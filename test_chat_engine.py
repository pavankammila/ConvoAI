
from chat_engine import ChatEngine


class FakeLLM:
    """A simulated AI client for testing conversation memory."""

    def generate(self, system_prompt, messages):
        latest = messages[-1]["text"]

        if "what is my name" in latest.lower():
            return "Your name is Pavan."

        if "what did i tell you first" in latest.lower():
            first_message = next(
                m["text"] for m in messages if m["role"] == "user"
            )
            return f"Your first message was: {first_message}"

        return f"Mock AI received: {latest}"


def test_conversation_memory():
    engine = ChatEngine(
        FakeLLM(),
        max_history=12,
        keep_recent=6
    )

    # First message
    first = engine.send(None, "My name is Pavan.")
    conversation_id = first["conversation_id"]

    # Second message in the SAME conversation
    second = engine.send(conversation_id, "What is my name?")

    # Third message in the SAME conversation
    third = engine.send(
        conversation_id,
        "What did I tell you first?"
    )

    assert second["reply"] == "Your name is Pavan."
    assert "My name is Pavan." in third["reply"]

    print("PASS: Conversation memory works!")
    print("Conversation ID:", conversation_id)
    print("First response:", first["reply"])
    print("Second response:", second["reply"])
    print("Third response:", third["reply"])


if __name__ == "__main__":
    test_conversation_memory()
