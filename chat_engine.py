
"""
ConvoAI - Chat Engine

Responsibilities:
1. Maintain separate conversations.
2. Preserve multi-turn conversation history.
3. Persist conversations using SQLite.
4. Manage context window limits.
5. Summarize older messages.
6. Recover gracefully if summarization fails.
7. Restore conversations after application restarts.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid

from collections import OrderedDict
from dataclasses import dataclass, field

from errors import ChatError


log = logging.getLogger(__name__)


# ------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------

SYSTEM_PROMPT = """You are ConvoAI, a friendly and patient AI assistant.

Personality:
- Warm, encouraging and supportive.
- Never condescending.
- Explain concepts clearly.

How you help:
- Answer questions accurately and clearly.
- Explain complex concepts in simple language.
- Provide practical examples when useful.
- When users share errors, explain the cause and solution.
- Ask a clarifying question when necessary.

Constraints:
- Help with programming, computer science, education and general topics.
- Keep replies concise unless more detail is requested.
- Never invent technical information.
- Use Markdown for readable responses.
"""


SUMMARY_SYSTEM = """
You are a conversation summarization assistant.

Create a concise factual summary of the conversation.
Preserve:
- User goals and important facts.
- Previous decisions.
- Important technical details.
- Unresolved questions.

Maximum length: 150 words.
Output only the summary.
"""


# ------------------------------------------
# CONFIGURATION
# ------------------------------------------

MAX_MESSAGE_CHARS = 4000
MAX_CONVERSATIONS = 200


# ------------------------------------------
# CONVERSATION DATA MODEL
# ------------------------------------------

@dataclass
class Conversation:
    history: list[dict] = field(default_factory=list)
    summary: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)


# ------------------------------------------
# CHAT ENGINE
# ------------------------------------------

class ChatEngine:

    def __init__(
        self,
        llm,
        system_prompt: str = SYSTEM_PROMPT,
        max_history: int | None = None,
        keep_recent: int | None = None,
        database=None,
    ):

        self.llm = llm
        self.system_prompt = system_prompt
        self.database = database

        self.max_history = (
            max_history
            if max_history is not None
            else int(os.getenv("MAX_HISTORY", "12"))
        )

        self.keep_recent = (
            keep_recent
            if keep_recent is not None
            else int(os.getenv("KEEP_RECENT", "6"))
        )

        if self.keep_recent < 2 or self.keep_recent >= self.max_history:
            raise ValueError("Need 2 <= KEEP_RECENT < MAX_HISTORY")

        self._conversations = OrderedDict()
        self._store_lock = threading.Lock()

    # --------------------------------------
    # SEND MESSAGE
    # --------------------------------------

    def send(self, conversation_id: str | None, text: str) -> dict:

        text = (text or "").strip()

        if not text:
            raise ChatError("Please type a message first.", 400)

        if len(text) > MAX_MESSAGE_CHARS:
            raise ChatError(
                f"Message is too long. Maximum {MAX_MESSAGE_CHARS} characters.",
                400
            )

        conversation_id, conv = self._get_or_create(conversation_id)

        with conv.lock:

            # Keep a snapshot so failed requests can be rolled back.
            previous_history = list(conv.history)
            previous_summary = conv.summary

            conv.history.append({
                "role": "user",
                "text": text
            })

            try:
                self._compact(conv)

                reply = self.llm.generate(
                    self._system_for(conv),
                    list(conv.history)
                )

                if not isinstance(reply, str) or not reply.strip():
                    raise ChatError(
                        "The AI returned an empty response. Please try again.",
                        502
                    )

                reply = reply.strip()

            except Exception:
                conv.history = previous_history
                conv.summary = previous_summary
                raise

            conv.history.append({
                "role": "model",
                "text": reply
            })

            # Save active history and summary together.
            if self.database is not None:
                try:
                    self.database.save_conversation_state(
                        conversation_id,
                        conv.history,
                        conv.summary
                    )

                except Exception as error:
                    conv.history = previous_history
                    conv.summary = previous_summary

                    log.exception("Failed to persist conversation state")

                    raise ChatError(
                        "Your response was generated, but saving the conversation failed.",
                        500
                    ) from error

            return {
                "conversation_id": conversation_id,
                "reply": reply,
                "messages_in_context": len(conv.history),
                "has_summary": bool(conv.summary)
            }

    # --------------------------------------
    # RESET CONVERSATION
    # --------------------------------------

    def reset(self, conversation_id: str | None) -> None:

        if not conversation_id:
            return

        with self._store_lock:
            self._conversations.pop(conversation_id, None)

        if self.database is not None:
            self.database.delete_conversation(conversation_id)

    # --------------------------------------
    # CREATE OR RETRIEVE CONVERSATION
    # --------------------------------------

    def _get_or_create(self, conversation_id: str | None):

        with self._store_lock:

            if conversation_id and conversation_id in self._conversations:
                self._conversations.move_to_end(conversation_id)

                return conversation_id, self._conversations[conversation_id]

            new_id = conversation_id or uuid.uuid4().hex

            conv = Conversation()

            # Restore an existing conversation from SQLite.
            if self.database is not None:

                saved = self.database.get_conversation(new_id)

                if saved:

                    conv.summary = saved["summary"]

                    saved_messages = self.database.get_messages(new_id)

                    conv.history = [
                        {
                            "role": message["role"],
                            "text": message["content"]
                        }
                        for message in saved_messages
                    ]

                else:
                    self.database.create_conversation(new_id)

            self._conversations[new_id] = conv

            while len(self._conversations) > MAX_CONVERSATIONS:
                self._conversations.popitem(last=False)

            return new_id, conv

    # --------------------------------------
    # SYSTEM PROMPT WITH MEMORY
    # --------------------------------------

    def _system_for(self, conv: Conversation) -> str:

        if not conv.summary:
            return self.system_prompt

        return (
            f"{self.system_prompt}\n\n"
            "Earlier conversation summary:\n"
            f"{conv.summary}\n"
            "Use this summary as background context."
        )

    # --------------------------------------
    # CONTEXT WINDOW MANAGEMENT
    # --------------------------------------

    def _compact(self, conv: Conversation) -> None:

        if len(conv.history) <= self.max_history:
            return

        cut = len(conv.history) - self.keep_recent

        old = conv.history[:cut]
        recent = conv.history[cut:]

        # Ensure recent history starts with a user message.
        while recent and recent[0]["role"] != "user":
            old.append(recent.pop(0))

        try:
            new_summary = self._summarize(
                conv.summary,
                old
            )

            conv.summary = new_summary

        except Exception as error:

            log.warning(
                "Summarization failed. Falling back to truncation: %s",
                error
            )

        # Retain recent messages whether summarization succeeds or fails.
        # The updated active history is persisted after successful generation.
        conv.history = recent

    # --------------------------------------
    # SUMMARIZE OLDER MESSAGES
    # --------------------------------------

    def _summarize(self, previous_summary: str, old: list[dict]) -> str:

        lines = []

        for message in old:

            role = (
                "User"
                if message["role"] == "user"
                else "Assistant"
            )

            lines.append(
                f"{role}: {message['text'][:1000]}"
            )

        prompt = (
            f"Previous summary:\n{previous_summary or '(none)'}\n\n"
            "New conversation messages:\n"
            + "\n".join(lines)
            + "\n\nCreate an updated summary."
        )

        result = self.llm.generate(
            SUMMARY_SYSTEM,
            [{"role": "user", "text": prompt}]
        )

        if not isinstance(result, str) or not result.strip():
            raise ValueError("Empty summary returned by AI.")

        return result.strip()[:2000]
