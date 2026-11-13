import test from "node:test";
import assert from "node:assert/strict";
import { ChatEngine, MAX_CONVERSATIONS } from "../src/chat_engine.js";
import { Database, type HistoryMessage } from "../src/database.js";
import { ChatError } from "../src/errors.js";
import type { GenerateOptions, LLMClient } from "../src/llm_client.js";

class MockLLM implements LLMClient {
  public calls: Array<{ system: string; messages: HistoryMessage[] }> = [];
  public failMain = false;
  public failSummary = false;

  async generate(
    system: string,
    messages: HistoryMessage[],
    _options?: GenerateOptions
  ): Promise<string> {
    this.calls.push({ system, messages });

    if (system.includes("summarization assistant")) {
      if (this.failSummary) {
        throw new ChatError("Summary failed", 503);
      }
      return "Summary: discussed topics A and B.";
    }

    if (this.failMain) {
      throw new ChatError("Provider unavailable", 503);
    }

    const last = messages[messages.length - 1]?.text ?? "";
    return `Reply to: ${last}`;
  }

  async generateStream(
    system: string,
    messages: HistoryMessage[],
    onChunk: (chunk: string) => void,
    options?: GenerateOptions
  ): Promise<string> {
    const reply = await this.generate(system, messages, options);
    const mid = Math.ceil(reply.length / 2);
    onChunk(reply.slice(0, mid));
    onChunk(reply.slice(mid));
    return reply;
  }
}

test("ChatEngine preserves multi-turn history, compacts with summary, and falls back to truncation", async () => {
  const db = new Database(":memory:");
  const llm = new MockLLM();
  const engine = new ChatEngine(llm, {
    maxHistory: 4,
    keepRecent: 2,
    database: db,
  });

  try {
    // Turn 1 (history -> 2)
    const r1 = await engine.send("c1", "First question");
    assert.equal(r1.messages_in_context, 2);
    assert.equal(r1.has_summary, false);

    // Turn 2 (history -> 4)
    const r2 = await engine.send("c1", "Second question");
    assert.equal(r2.messages_in_context, 4);
    assert.equal(r2.has_summary, false);

    // Turn 3 (pushes to 5 before LLM -> compacts [u1,m1,u2,m2] into summary, keeps [u3] + reply -> 2)
    const r3 = await engine.send("c1", "Third question");
    assert.equal(r3.has_summary, true);
    assert.equal(r3.messages_in_context, 2);

    // Full display transcript in SQLite still has all 6 messages!
    assert.equal(db.getMessages("c1").length, 6);
    assert.equal(db.getContextMessages("c1").length, 2);

    // Turn 4 (len 2 -> 3 -> 4, no compaction yet)
    const r4 = await engine.send("c1", "Fourth question");
    assert.equal(r4.messages_in_context, 4);

    // Clear summary and verify truncation fallback if summary fails on Turn 5
    assert.equal(engine.clearSummary("c1"), true);
    llm.failSummary = true;
    const r5 = await engine.send("c1", "Fifth question");
    assert.equal(r5.has_summary, false);
    assert.equal(r5.messages_in_context, 2);
    assert.equal(db.getMessages("c1").length, 10);
  } finally {
    db.close();
  }
});

test("ChatEngine rolls back state on LLM failure and restores from SQLite after cache eviction", async () => {
  const db = new Database(":memory:");
  const llm = new MockLLM();
  const engine = new ChatEngine(llm, {
    maxHistory: 6,
    keepRecent: 2,
    database: db,
  });

  try {
    await engine.send("restore-me", "Remember my favorite color is orange.");
    llm.failMain = true;
    await assert.rejects(() => engine.send("restore-me", "This turn fails"), ChatError);
    assert.equal(db.getMessages("restore-me").length, 2);

    // Evict from in-memory LRU cache and ensure SQLite restores context
    llm.failMain = false;
    engine.invalidateCache("restore-me");

    const chunks: string[] = [];
    const streamed = await engine.sendStream("restore-me", "What did I say?", {
      onChunk: (c) => chunks.push(c),
    });
    assert.equal(chunks.join(""), streamed.reply);
    assert.equal(streamed.messages_in_context, 4);

    // Regenerate last turn
    const regenChunks: string[] = [];
    const regen = await engine.regenerateStream("restore-me", {
      onChunk: (c) => regenChunks.push(c),
    });
    assert.equal(regen.reply, "Reply to: What did I say?");
    assert.equal(regenChunks.join(""), regen.reply);
    assert.equal(db.getMessages("restore-me").length, 4);
    assert.ok(MAX_CONVERSATIONS >= 200);
  } finally {
    db.close();
  }
});

test("ChatEngine handles Pavan multi-turn recall, full SQLite restart restoration, and zero duplicate history on retry", async () => {
  const os = await import("node:os");
  const path = await import("node:path");
  const fs = await import("node:fs");
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "convoai-restart-"));
  const dbFile = path.join(tmpDir, "test.db");

  class MemoryAwareMock implements LLMClient {
    public failNext = false;
    async generate(system: string, messages: HistoryMessage[]): Promise<string> {
      if (this.failNext) {
        this.failNext = false;
        throw new ChatError("Temporary outage", 503);
      }
      if (system.includes("summarization assistant")) {
        return "User's name is Pavan.";
      }
      const combined = messages.map((m) => m.text).join("\n") + "\n" + system;
      const last = messages[messages.length - 1]?.text ?? "";
      if (/what is my name/i.test(last)) {
        const match = combined.match(/(?:My|User's) name is ([A-Za-z]+)/i);
        return match ? `Your name is ${match[1]}.` : "I do not know your name.";
      }
      if (/My name is Pavan/i.test(last)) {
        return "Nice to meet you, Pavan!";
      }
      return `Ack: ${last}`;
    }
  }

  const llm1 = new MemoryAwareMock();
  const db1 = new Database(dbFile);
  try {
    const engine1 = new ChatEngine(llm1, {
      maxHistory: 4,
      keepRecent: 2,
      database: db1,
    });

    const t1 = await engine1.send("pavan-chat", "My name is Pavan.");
    assert.equal(t1.reply, "Nice to meet you, Pavan!");

    // Trigger failure then retry same message; history must NOT duplicate
    llm1.failNext = true;
    await assert.rejects(() => engine1.send("pavan-chat", "What is my name?"), ChatError);
    assert.equal(db1.getMessages("pavan-chat").length, 2);

    const t2 = await engine1.send("pavan-chat", "What is my name?");
    assert.equal(t2.reply, "Your name is Pavan.");
    assert.equal(db1.getMessages("pavan-chat").length, 4);

    // Compact into summary on turn 3
    await engine1.send("pavan-chat", "Let's discuss TypeScript.");
  } finally {
    db1.close();
  }

  // Simulate full application restart with fresh Database + ChatEngine instance
  const db2 = new Database(dbFile);
  const llm2 = new MemoryAwareMock();
  try {
    const engine2 = new ChatEngine(llm2, {
      maxHistory: 4,
      keepRecent: 2,
      database: db2,
    });
    const t4 = await engine2.send("pavan-chat", "What is my name?");
    assert.equal(t4.reply, "Your name is Pavan.");
    assert.equal(t4.has_summary, true);
  } finally {
    db2.close();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test("ChatEngine applies personalities, length, tone, per-conversation instructions, and separates short-term vs explicit long-term memory", async () => {
  const db = new Database(":memory:");
  const seenSystems: string[] = [];

  class InspectorLLM implements LLMClient {
    async generate(system: string, messages: HistoryMessage[]): Promise<string> {
      seenSystems.push(system);
      if (system.includes("analyze a conversation and extract ONLY durable")) {
        return JSON.stringify([
          { category: "preference", content: "Prefers TypeScript with Express" },
          { category: "general", content: "My API key is AIzaSyFakeKey1234567890123" },
        ]);
      }
      return `Answered (${messages.length} msgs)`;
    }
  }

  const engine = new ChatEngine(new InspectorLLM(), { database: db });

  try {
    // 1. Default (no consent): no personal preferences or memories injected
    await engine.send("chat-a", "Remember that my secret project is Apollo.");
    assert.match(seenSystems[0], /Friendly Assistant/);
    assert.match(seenSystems[0], /Balanced/);
    assert.match(seenSystems[0], /Neutral/);
    assert.doesNotMatch(seenSystems[0], /Explicit Long-Term User Memory \(facts/);

    // 2. Conversation B does NOT receive Chat A's short-term history!
    await engine.send("chat-b", "Do you know my project?");
    assert.doesNotMatch(seenSystems[1], /Apollo/);

    // 3. Grant consent + set Coding personality, Concise length, Direct tone + Custom & Memory
    db.savePreferences({
      consent_given: true,
      preferred_name: "Pavan",
      personality: "coding",
      response_length: "concise",
      response_tone: "direct",
      memory_enabled: true,
    });
    db.addMemory("Uses SQLite with Node.js", "preference");

    // Conversation A gets Coding + Concise + Direct + Preferred Name + Explicit Long-Term Memory
    await engine.send("chat-a", "Write a quick SQLite helper.");
    const sysCoding = seenSystems[2];
    assert.match(sysCoding, /Coding Assistant/);
    assert.match(sysCoding, /Concise/);
    assert.match(sysCoding, /Direct/);
    assert.match(sysCoding, /User's preferred name: Pavan/);
    assert.match(sysCoding, /Uses SQLite with Node\.js/);

    // 4. Custom system personality
    db.savePreferences({
      consent_given: true,
      personality: "custom",
      custom_system_prompt: "You are a strict Socratic mentor who asks guiding questions.",
    });
    await engine.send("chat-b", "Help me learn recursion.");
    assert.match(seenSystems[3], /strict Socratic mentor/);

    // 5. Conversation-specific instructions & per-chat personality override (isolated to chat-b)
    db.updateConversationSettings("chat-b", {
      custom_instructions: "Respond using Python analogies only.",
      personality: "study",
      response_length: "detailed",
      response_tone: "warm",
    });
    engine.syncConversationSettings("chat-b", {
      custom_instructions: "Respond using Python analogies only.",
      personality: "study",
      response_length: "detailed",
      response_tone: "warm",
    });

    await engine.send("chat-b", "Explain call stacks.");
    const sysStudy = seenSystems[4];
    assert.match(sysStudy, /Study Assistant/);
    assert.match(sysStudy, /Detailed/);
    assert.match(sysStudy, /Warm/);
    assert.match(sysStudy, /Respond using Python analogies only\./);

    // Chat A should NOT receive Chat B's conversation-specific instructions
    await engine.send("chat-a", "Another question.");
    assert.doesNotMatch(seenSystems[5], /Python analogies only/);

    // 6. Extract useful memories filters out sensitive API key automatically
    const extracted = await engine.extractUsefulMemories("chat-a");
    assert.equal(extracted.added.length, 1);
    assert.equal(extracted.skippedSensitive, 1);
    assert.equal(extracted.added[0].content, "Prefers TypeScript with Express");
  } finally {
    db.close();
  }
});

test("ChatEngine passes multimodal attachments, preserves them across regenerate and compaction summaries, and propagates consent-backed model selection", async () => {
  const db = new Database(":memory:");
  const recordedCalls: Array<{
    system: string;
    messages: HistoryMessage[];
    options?: GenerateOptions;
  }> = [];

  class MultimodalMock implements LLMClient {
    getModel() {
      return "gemini-4-hydrogen";
    }
    async generate(
      system: string,
      messages: HistoryMessage[],
      options?: GenerateOptions
    ): Promise<string> {
      recordedCalls.push({ system, messages, options });
      if (system.includes("summarization assistant")) {
        return "Summary: user shared an image.";
      }
      return `Got ${messages.length} msgs`;
    }
    async generateStream(
      system: string,
      messages: HistoryMessage[],
      onChunk: (chunk: string) => void,
      options?: GenerateOptions
    ): Promise<string> {
      const reply = await this.generate(system, messages, options);
      onChunk(reply);
      return reply;
    }
  }

  const engine = new ChatEngine(new MultimodalMock(), {
    maxHistory: 4,
    keepRecent: 2,
    database: db,
  });

  try {
    db.savePreferences({
      consent_given: true,
      model: "gemini-3.1-flash-lite",
    });

    const imgAtt = [
      {
        kind: "image" as const,
        name: "photo.png",
        mimeType: "image/png",
        data: "BASE64PAYLOADSECRET",
      },
    ];

    // Turn 1 with image attachment (empty prompt uses fallback text)
    const r1 = await engine.send("mm-chat", "", imgAtt);
    assert.equal(r1.reply, "Got 1 msgs");
    assert.equal(recordedCalls[0].options?.model, "gemini-3.1-flash-lite");
    assert.equal(recordedCalls[0].messages[0].attachments?.[0].name, "photo.png");

    // Regenerate preserves attachment after cache eviction
    engine.invalidateCache("mm-chat");
    const regen = await engine.regenerateStream("mm-chat", { onChunk: () => {} });
    assert.equal(regen.reply, "Got 1 msgs");
    assert.equal(recordedCalls[1].messages[0].attachments?.[0].name, "photo.png");

    // Turn 2 & Turn 3 trigger compaction; summary prompt must use [Attached: photo.png], not raw base64
    await engine.send("mm-chat", "Follow up one");
    await engine.send("mm-chat", "Follow up two");
    const summaryCall = recordedCalls.find((c) =>
      c.system.includes("summarization assistant")
    );
    assert.ok(summaryCall);
    assert.match(summaryCall.messages[0].text, /\[Attached: photo\.png\]/);
    assert.doesNotMatch(summaryCall.messages[0].text, /BASE64PAYLOADSECRET/);
  } finally {
    db.close();
  }
});

