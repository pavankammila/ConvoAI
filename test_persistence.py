
from pathlib import Path

from database import Database
from chat_engine import ChatEngine


class FakeLLM:
    """Mock LLM that works without calling the Gemini API."""

    def __init__(self):
        self.last_messages = []

    def generate(self, system_prompt, messages):
        # Save the messages received by the mock AI
        self.last_messages = list(messages)

        return "Mock AI response received successfully."


def main():
    print("=" * 50)
    print("CONVOAI - DATABASE PERSISTENCE TEST")
    print("=" * 50)

    # Start with a fresh test database
    db_path = Path("test_persistence.db")

    if db_path.exists():
        db_path.unlink()

    db = Database(str(db_path))

    # Create the first ChatEngine
    fake_llm = FakeLLM()

    engine = ChatEngine(
        llm=fake_llm,
        database=db
    )

    # -----------------------------------------
    # TEST 1: Create a conversation
    # -----------------------------------------

    print("\nTEST 1: Creating a conversation...")

    result = engine.send(
        None,
        "My name is Pavan."
    )

    conversation_id = result["conversation_id"]

    print("PASS: Conversation created")
    print("Conversation ID:", conversation_id)
    print("AI response:", result["reply"])

    # -----------------------------------------
    # TEST 2: Check database storage
    # -----------------------------------------

    print("\nTEST 2: Checking database storage...")

    saved_messages = db.get_messages(conversation_id)

    for message in saved_messages:
        print(message)

    if len(saved_messages) >= 2:
        print("PASS: Messages stored in SQLite")
    else:
        print("FAIL: Expected user and AI messages")

    # -----------------------------------------
    # TEST 3: Simulate application restart
    # -----------------------------------------

    print("\nTEST 3: Restoring conversation...")

    fake_llm_after_restart = FakeLLM()

    engine_after_restart = ChatEngine(
        llm=fake_llm_after_restart,
        database=db
    )

    result_after_restart = engine_after_restart.send(
        conversation_id,
        "What was my first message?"
    )

    print("AI response:", result_after_restart["reply"])

    # -----------------------------------------
    # TEST 4: Verify previous context restored
    # -----------------------------------------

    print("\nTEST 4: Checking restored context...")

    restored_messages = fake_llm_after_restart.last_messages

    print("Messages received by AI after restart:")

    for message in restored_messages:
        print(message)

    previous_context_found = (
        "My name is Pavan." in repr(restored_messages)
    )

    if previous_context_found:
        print("PASS: Previous conversation context restored")
    else:
        print("WARNING: Previous message not found in restored context")

    print("\n" + "=" * 50)
    print("PERSISTENCE TEST FINISHED")
    print("=" * 50)


if __name__ == "__main__":
    main()
