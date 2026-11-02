
"""
ConvoAI - Database Tests
"""

import os
from database import Database


TEST_DB = "test_convoai.db"


def test_database():

    # Remove previous test database if present.
    for suffix in ("", "-wal", "-shm"):
        path = TEST_DB + suffix

        if os.path.exists(path):
            os.remove(path)

    # Initialize test database.
    db = Database(TEST_DB)

    conversation_id = "test-conversation-001"

    # 1. Create conversation
    db.create_conversation(conversation_id)

    print("PASS: Database initialization")
    print("PASS: Conversation creation")

    # 2. Save messages
    db.add_message(
        conversation_id,
        "user",
        "My name is Pavan."
    )

    db.add_message(
        conversation_id,
        "model",
        "Nice to meet you, Pavan!"
    )

    print("PASS: Message storage")

    # 3. Retrieve messages
    messages = db.get_messages(conversation_id)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "My name is Pavan."
    assert messages[1]["content"] == "Nice to meet you, Pavan!"

    print("PASS: Message retrieval")

    # 4. Retrieve conversations
    conversations = db.get_conversations()

    assert len(conversations) == 1
    assert conversations[0]["id"] == conversation_id
    assert conversations[0]["title"] == "My name is Pavan."

    print("PASS: Conversation listing")

    # 5. Delete conversation
    deleted = db.delete_conversation(conversation_id)

    assert deleted is True
    assert db.get_messages(conversation_id) == []
    assert db.get_conversations() == []

    print("PASS: Conversation deletion")

    # 6. Confirm nonexistent conversation deletion
    assert db.delete_conversation("does-not-exist") is False

    print("PASS: Missing conversation handling")

    # Close and remove test database files.
    for suffix in ("", "-wal", "-shm"):
        path = TEST_DB + suffix

        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    test_database()
    print("\nAll database tests completed successfully!")
