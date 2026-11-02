
"""
ConvoAI - SQLite Database Manager

Responsibilities:
1. Store conversations permanently.
2. Store user and assistant messages.
3. Store conversation summaries.
4. Retrieve conversation history.
5. List and delete conversations.
6. Save compacted conversation state atomically.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager


DATABASE_PATH = Path(__file__).resolve().parent / "convoai.db"


class Database:

    def __init__(self, database_path=DATABASE_PATH):
        self.database_path = str(database_path)
        self.initialize()

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.database_path, timeout=10)
        conn.row_factory = sqlite3.Row

        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def initialize(self):
        """Create tables and safely migrate older database schemas."""

        with self.connection() as conn:

            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New Chat',
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL
                        CHECK(role IN ('user', 'model')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,

                    FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
                )
            """)

            columns = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(conversations)"
                ).fetchall()
            }

            if "summary" not in columns:
                conn.execute("""
                    ALTER TABLE conversations
                    ADD COLUMN summary TEXT NOT NULL DEFAULT ''
                """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_messages_conversation
                ON messages(conversation_id, id)
            """)

    def create_conversation(self, conversation_id, title="New Chat"):
        now = self.now()

        with self.connection() as conn:
            conn.execute("""
                INSERT INTO conversations
                (id, title, summary, created_at, updated_at)
                VALUES (?, ?, '', ?, ?)
            """, (conversation_id, title, now, now))

    def add_message(self, conversation_id, role, content):
        """Add one message. Retained for compatibility with existing tests."""

        if role not in ("user", "model"):
            raise ValueError("Invalid message role.")

        now = self.now()

        with self.connection() as conn:
            cursor = conn.execute("""
                INSERT INTO messages
                (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, role, content, now))

            if cursor.rowcount != 1:
                raise RuntimeError("Message could not be saved.")

            if role == "user":
                conn.execute("""
                    UPDATE conversations
                    SET title = CASE
                        WHEN title = 'New Chat' THEN ?
                        ELSE title
                    END,
                    updated_at = ?
                    WHERE id = ?
                """, (content.strip()[:50] or "New Chat", now, conversation_id))
            else:
                conn.execute("""
                    UPDATE conversations
                    SET updated_at = ?
                    WHERE id = ?
                """, (now, conversation_id))

    def save_conversation_state(self, conversation_id, messages, summary):
        """
        Atomically replace the stored active history and summary.

        Older messages removed during context compaction are deleted
        from SQLite because their information is represented by summary.
        """

        now = self.now()

        with self.connection() as conn:

            conversation = conn.execute("""
                SELECT id, title
                FROM conversations
                WHERE id = ?
            """, (conversation_id,)).fetchone()

            if conversation is None:
                raise ValueError("Conversation does not exist.")

            # Remove previously stored active messages.
            conn.execute("""
                DELETE FROM messages
                WHERE conversation_id = ?
            """, (conversation_id,))

            # Store only the current active context.
            for message in messages:

                role = message.get("role")
                content = message.get("text", "")

                if role not in ("user", "model"):
                    raise ValueError("Invalid message role.")

                conn.execute("""
                    INSERT INTO messages
                    (conversation_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                """, (conversation_id, role, content, now))

            # Use the first user message as the conversation title.
            title = conversation["title"]

            if title == "New Chat":
                for message in messages:
                    if message.get("role") == "user":
                        title = message.get("text", "").strip()[:50]
                        title = title or "New Chat"
                        break

            conn.execute("""
                UPDATE conversations
                SET title = ?,
                    summary = ?,
                    updated_at = ?
                WHERE id = ?
            """, (title, summary or "", now, conversation_id))

    def get_messages(self, conversation_id):
        with self.connection() as conn:
            rows = conn.execute("""
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
            """, (conversation_id,)).fetchall()

            return [dict(row) for row in rows]

    def get_conversation(self, conversation_id):
        with self.connection() as conn:
            row = conn.execute("""
                SELECT id, title, summary, created_at, updated_at
                FROM conversations
                WHERE id = ?
            """, (conversation_id,)).fetchone()

            return dict(row) if row else None

    def update_summary(self, conversation_id, summary):
        """Update a conversation summary."""

        with self.connection() as conn:
            cursor = conn.execute("""
                UPDATE conversations
                SET summary = ?, updated_at = ?
                WHERE id = ?
            """, (summary, self.now(), conversation_id))

            return cursor.rowcount > 0

    def get_conversations(self):
        with self.connection() as conn:
            rows = conn.execute("""
                SELECT id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
            """).fetchall()

            return [dict(row) for row in rows]

    def delete_conversation(self, conversation_id):
        with self.connection() as conn:
            cursor = conn.execute("""
                DELETE FROM conversations
                WHERE id = ?
            """, (conversation_id,))

            return cursor.rowcount > 0

    def clear_database(self):
        """Clear all conversations. For testing only."""

        with self.connection() as conn:
            conn.execute("DELETE FROM conversations")
