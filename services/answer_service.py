"""
Answer Service — Groq Cloud (LLaMA via Groq API)
=================================================
Takes an English question and a list of retrieved context chunks,
builds a grounded prompt, calls the Groq API, and returns an
English answer string.

The service is intentionally stateless: every call is independent,
which keeps it easy to plug into any RAG pipeline step.

Usage:
    from services.answer_service import AnswerService

    svc = AnswerService()                          # reads GROQ_API_KEY from env
    answer = svc.answer(
        question="What documents do I need to register with CNSS?",
        chunks=["Chunk A …", "Chunk B …"],         # retrieved context (can be [])
    )
"""

import logging
import os
from typing import Optional

from groq import Groq, APIError, AuthenticationError, RateLimitError
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL       = "llama-3.3-70b-versatile"
DEFAULT_MAX_TOKENS  = 1024
DEFAULT_TEMPERATURE = 0.2   # low temp → factual, consistent answers

_SYSTEM_PROMPT = """\
You are DarijaAssist, an intelligent assistant that answers questions about \
Moroccan administrative procedures and daily life topics.

You will be given a user question in English and, optionally, a set of \
context passages retrieved from a knowledge base. Use the context whenever \
it is relevant. If the context does not contain enough information, answer \
from your general knowledge but be transparent about it.

Always respond in clear, concise English.\
"""

_CONTEXT_HEADER = "### Retrieved Context\n"
_NO_CONTEXT_MSG = "(No additional context was provided.)"


class AnswerService:
    """
    Stateless wrapper around the Groq chat-completions API.

    Initialises the Groq client once with the API key, then exposes a single
    `answer(question, chunks)` method that builds a grounded prompt and
    returns the model's response as a plain string.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Configure the Groq client.

        Args:
            api_key:           Groq Cloud API key.
                               Falls back to the ``GROQ_API_KEY`` env var
                               (loaded from .env automatically).
            model_name:        Model hosted on Groq (e.g. llama-3.3-70b-versatile).
            max_output_tokens: Maximum tokens in the generated answer.
            temperature:       Sampling temperature (0 = deterministic).

        Raises:
            EnvironmentError: If no API key is found.
            RuntimeError:     If the Groq client fails to initialise.
        """
        load_dotenv()

        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "No Groq API key found. "
                "Set GROQ_API_KEY in your .env file or pass it explicitly."
            )

        self._model_name        = model_name
        self._max_output_tokens = max_output_tokens
        self._temperature       = temperature

        try:
            self._client = Groq(api_key=resolved_key)
            logger.info("✅ AnswerService ready — model: %s", self._model_name)
        except Exception as exc:
            logger.error("❌ Failed to initialise Groq client: %s", exc)
            raise RuntimeError(
                "Could not initialise the Groq client. "
                "Check your API key and network connection."
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        chunks: Optional[list[str]] = None,
    ) -> str:
        """
        Generate an English answer grounded in the supplied context chunks.

        Args:
            question: English question to answer.
            chunks:   List of retrieved context strings (may be empty or None).
                      In the full pipeline these come from the RAG retriever;
                      for now they are passed in as a plain parameter.

        Returns:
            English answer as a plain string.

        Raises:
            ValueError:  If the question is empty.
            RuntimeError: If the Groq API call fails.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        user_prompt = self._build_prompt(question, chunks or [])
        logger.debug("Groq prompt (truncated): %s …", user_prompt[:200])

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=self._max_output_tokens,
                temperature=self._temperature,
            )

            answer_text = response.choices[0].message.content
            if not answer_text:
                raise RuntimeError(
                    "Groq returned an empty response. "
                    "The content may have been filtered."
                )

            answer_text = answer_text.strip()
            logger.debug("Groq answer (truncated): %s …", answer_text[:200])
            return answer_text

        except (ValueError, RuntimeError):
            raise
        except AuthenticationError as exc:
            logger.error("❌ Groq auth error: %s", exc)
            raise RuntimeError(
                "Groq authentication failed — check your API key."
            ) from exc
        except RateLimitError as exc:
            logger.error("❌ Groq rate limit: %s", exc)
            raise RuntimeError(
                "Groq rate limit exceeded — wait and retry."
            ) from exc
        except APIError as exc:
            logger.error("❌ Groq API error [%s]: %s", exc.status_code, exc)
            raise RuntimeError(
                f"Groq API error ({exc.status_code}): {exc.message}"
            ) from exc
        except Exception as exc:
            logger.error("❌ Unexpected error: %s", exc)
            raise RuntimeError(
                f"Groq API call failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(question: str, chunks: list[str]) -> str:
        """
        Assemble the full user-turn prompt from the question and chunks.

        Format:
            ### Retrieved Context
            [1] chunk text …
            [2] chunk text …
            …

            ### Question
            <question>
        """
        lines: list[str] = [_CONTEXT_HEADER]

        if chunks:
            for i, chunk in enumerate(chunks, start=1):
                lines.append(f"[{i}] {chunk.strip()}")
        else:
            lines.append(_NO_CONTEXT_MSG)

        lines.append("\n### Question")
        lines.append(question.strip())

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the model identifier in use."""
        return self._model_name

    def __repr__(self) -> str:
        return f"<AnswerService model='{self._model_name}'>"
