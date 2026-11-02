
from chat_engine import ChatEngine, SUMMARY_SYSTEM


class FailingSummaryLLM:
    """Simulates successful chat but failed summarization."""

    def __init__(self):
        self.summary_attempts = 0

    def generate(self, system_prompt, messages):

        if system_prompt == SUMMARY_SYSTEM:
            self.summary_attempts += 1
            raise RuntimeError("Simulated summarization failure")

        return f"Mock response: {messages[-1]['text']}"


def test_summarization_fallback():

    fake_llm = FailingSummaryLLM()

    engine = ChatEngine(
        fake_llm,
        max_history=12,
        keep_recent=6
    )

    conversation_id = None

    # Send seven messages to trigger compaction
    for i in range(1, 8):
        result = engine.send(
            conversation_id,
            f"Test message {i}"
        )
        conversation_id = result["conversation_id"]

    conversation = engine._conversations[conversation_id]

    assert fake_llm.summary_attempts >= 1
    assert conversation.summary == ""
    assert len(conversation.history) <= 6

    print("PASS: Summarization fallback works!")
    print("Summary attempts:", fake_llm.summary_attempts)
    print("Summary stored:", bool(conversation.summary))
    print("Messages retained:", len(conversation.history))
    print("Latest message:", conversation.history[-2]["text"])


if __name__ == "__main__":
    test_summarization_fallback()
