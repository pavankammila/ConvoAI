
from chat_engine import ChatEngine, SUMMARY_SYSTEM


class FakeLLM:
    """Simulated LLM that records summary requests."""

    def __init__(self):
        self.summary_calls = 0

    def generate(self, system_prompt, messages):
        # Simulate the summarization request
        if system_prompt == SUMMARY_SYSTEM:
            self.summary_calls += 1
            return "Summary: The user is testing ConvoAI context management."

        # Simulate an ordinary chat response
        latest = messages[-1]["text"]
        return f"Mock response to: {latest}"


def test_context_management():
    fake_llm = FakeLLM()

    engine = ChatEngine(
        fake_llm,
        max_history=12,
        keep_recent=6
    )

    conversation_id = None

    # Send enough messages to trigger summarization
    for i in range(1, 9):
        result = engine.send(
            conversation_id,
            f"This is test message number {i}."
        )
        conversation_id = result["conversation_id"]

    # Inspect the conversation
    conversation = engine._conversations[conversation_id]

    assert fake_llm.summary_calls >= 1
    assert conversation.summary != ""
    assert len(conversation.history) <= 12

    print("PASS: Context management works!")
    print("Summary calls:", fake_llm.summary_calls)
    print("Summary:", conversation.summary)
    print("Messages currently retained:", len(conversation.history))
    print("Recent messages retained:", conversation.history)


if __name__ == "__main__":
    test_context_management()
