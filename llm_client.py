
"""
ConvoAI - Gemini API Client

The only module that communicates with Google Gemini.

Features:
- Gemini 3.8 Flash integration
- Secure API key handling
- Request timeouts
- Automatic retries
- Exponential backoff
- Friendly API error messages
- Provider-independent interface
"""

from __future__ import annotations

import logging
import os
import time

from errors import ChatError

log = logging.getLogger(__name__)


# ------------------------------------------
# GEMINI CLIENT
# ------------------------------------------

class GeminiClient:

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash",
        timeout_s: float = 30,
        retries: int = 2,
        backoff_s: float = 1.0,
    ):

        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self.retries = retries
        self.backoff_s = backoff_s

        # Initialize Gemini client
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000)
            ),
        )

    # --------------------------------------
    # GENERATE AI RESPONSE
    # --------------------------------------

    def generate(self, system: str, messages: list[dict]) -> str:
        """
        Generate a response using Gemini.

        messages format:
        [
            {"role": "user", "text": "Hello"},
            {"role": "model", "text": "Hi there!"}
        ]
        """

        from google.genai import errors as genai_errors
        import httpx

        types = self._types

        # Convert conversation history into Gemini format
        contents = [
            types.Content(
                role=message["role"],
                parts=[
                    types.Part(text=message["text"])
                ]
            )
            for message in messages
        ]

        # Gemini 3.8 configuration
        # Temperature is intentionally omitted.
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1024,
        )

        last_error: ChatError | None = None

        # Retry failed requests
        for attempt in range(self.retries + 1):

            try:

                log.info(
                    "Sending request to Gemini model: %s",
                    self.model
                )

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )

                reply = (response.text or "").strip()

                if not reply:
                    raise ChatError(
                        "The AI returned an empty response. "
                        "Please try rephrasing your message.",
                        502
                    )

                log.info("Gemini response received successfully.")

                return reply

            except ChatError:
                raise

            except genai_errors.APIError as error:

                code = getattr(error, "code", None)

                log.warning(
                    "Gemini API error (attempt %d): %s",
                    attempt + 1,
                    error
                )

                last_error = _map_api_error(
                    code,
                    str(error)
                )

                # Retry only temporary errors
                if code not in (429, 500, 502, 503, 504):
                    raise last_error from error

            except httpx.TimeoutException:

                log.warning(
                    "Gemini request timed out (attempt %d)",
                    attempt + 1
                )

                last_error = ChatError(
                    "The AI took too long to respond. Please try again.",
                    504
                )

            except (httpx.HTTPError, OSError) as error:

                log.warning(
                    "Network error (attempt %d): %s",
                    attempt + 1,
                    error
                )

                last_error = ChatError(
                    "Couldn't reach the AI service. "
                    "Check your internet connection and try again.",
                    503
                )

            # Exponential backoff
            if attempt < self.retries:

                delay = self.backoff_s * (2 ** attempt)

                log.info(
                    "Retrying Gemini request in %s seconds...",
                    delay
                )

                time.sleep(delay)

        if last_error is not None:
            raise last_error

        raise ChatError(
            "Unable to generate an AI response. Please try again.",
            502
        )

    # --------------------------------------
    # CLOSE CLIENT
    # --------------------------------------

    def close(self):
        self.client.close()


# ------------------------------------------
# API ERROR MAPPING
# ------------------------------------------

def _map_api_error(
    code: int | None,
    message: str
) -> ChatError:

    lowered = message.lower()

    if code in (401, 403) or "api key" in lowered:

        return ChatError(
            "The API key was rejected. "
            "Check GEMINI_API_KEY in your .env file.",
            502
        )

    if code == 404:

        return ChatError(
            "The Gemini model was not found. "
            "Check your GEMINI_MODEL setting.",
            502
        )

    if code == 429:

        return ChatError(
            "Gemini rate limit reached. "
            "Please wait a few seconds and try again.",
            429
        )

    if code in (500, 502, 503, 504):

        return ChatError(
            "The Gemini service is temporarily busy. "
            "Please try again shortly.",
            503
        )

    return ChatError(
        "Something went wrong while communicating with Gemini.",
        502
    )


# ------------------------------------------
# UNCONFIGURED CLIENT
# ------------------------------------------

class UnconfiguredClient:
    """
    Allows the backend to start without an API key.
    Displays a friendly message when chat is attempted.
    """

    def generate(
        self,
        system: str,
        messages: list[dict]
    ) -> str:

        raise ChatError(
            "Gemini API key is missing. "
            "Please add GEMINI_API_KEY to your .env file.",
            503
        )


# ------------------------------------------
# CLIENT FACTORY
# ------------------------------------------

def build_client():

    api_key = os.getenv(
        "GEMINI_API_KEY",
        ""
    ).strip()

    if not api_key or api_key == "your-key-here":

        log.warning(
            "Gemini API key is not configured."
        )

        return UnconfiguredClient()

    return GeminiClient(
        api_key=api_key,
        model=os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        ),
        timeout_s=float(
            os.getenv("REQUEST_TIMEOUT", "30")
        ),
    )
