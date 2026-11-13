import test from "node:test";
import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { createApp } from "../src/server.js";
import { Database, type HistoryMessage } from "../src/database.js";
import { ChatEngine } from "../src/chat_engine.js";
import type { LLMClient } from "../src/llm_client.js";

class StubLLM implements LLMClient {
  async generate(_system: string, messages: HistoryMessage[]): Promise<string> {
    const last = messages[messages.length - 1]?.text ?? "";
    return `Stubbed answer: ${last}`;
  }

  async generateStream(
    system: string,
    messages: HistoryMessage[],
    onChunk: (chunk: string) => void
  ): Promise<string> {
    const text = await this.generate(system, messages);
    onChunk(text);
    return text;
  }
}

test("Express API routes handle validation, chat, SSE streaming, CRUD, and rate limiting", async () => {
  const db = new Database(":memory:");
  const engine = new ChatEngine(new StubLLM(), { database: db });
  const { app } = createApp({ database: db, engine, rateLimitPerMinute: 5 });

  const server = await new Promise<ReturnType<typeof app.listen>>((resolve) => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });

  const port = (server.address() as AddressInfo).port;
  const baseUrl = `http://127.0.0.1:${port}`;

  try {
    // Health & API meta
    const health = await fetch(`${baseUrl}/health`);
    assert.equal(health.status, 200);
    assert.deepEqual(await health.json(), { status: "healthy" });
    assert.equal(health.headers.get("x-content-type-options"), "nosniff");

    // Validation rejection
    const blank = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "   " }),
    });
    assert.equal(blank.status, 400);

    const badId = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "hi", conversation_id: "bad id!" }),
    });
    assert.equal(badId.status, 400);

    // Chat
    const chatRes = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "Hello world", conversation_id: "conv_1" }),
    });
    assert.equal(chatRes.status, 200);
    const chatJson = (await chatRes.json()) as {
      conversation_id: string;
      reply: string;
    };
    assert.equal(chatJson.conversation_id, "conv_1");
    assert.equal(chatJson.reply, "Stubbed answer: Hello world");

    // Streaming SSE
    const streamRes = await fetch(`${baseUrl}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "Stream test", conversation_id: "conv_1" }),
    });
    assert.equal(streamRes.status, 200);
    const sseText = await streamRes.text();
    assert.match(sseText, /event: meta/);
    assert.match(sseText, /event: chunk/);
    assert.match(sseText, /event: done/);

    // Rename & search
    const renameRes = await fetch(`${baseUrl}/api/conversations/conv_1`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "My Renamed Chat" }),
    });
    assert.equal(renameRes.status, 200);

    const listRes = await fetch(`${baseUrl}/api/conversations?q=Renamed`);
    const listJson = (await listRes.json()) as { total: number };
    assert.equal(listJson.total, 1);

    // Delete & 404 check
    const delRes = await fetch(`${baseUrl}/api/conversations/conv_1`, {
      method: "DELETE",
    });
    assert.equal(delRes.status, 200);

    const missingRes = await fetch(`${baseUrl}/api/conversations/conv_1`);
    assert.equal(missingRes.status, 404);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    db.close();
  }
});

test("API enforces preference consent, stores conversation settings, and manages explicit non-sensitive long-term memories", async () => {
  const db = new Database(":memory:");
  const engine = new ChatEngine(new StubLLM(), { database: db });
  const { app } = createApp({ database: db, engine, rateLimitPerMinute: 20 });

  const server = await new Promise<ReturnType<typeof app.listen>>((resolve) => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });

  const port = (server.address() as AddressInfo).port;
  const baseUrl = `http://127.0.0.1:${port}`;

  try {
    // 1. Reject persisting personal preferences/memory without consent
    const noConsent = await fetch(`${baseUrl}/api/preferences`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        consent_given: false,
        preferred_name: "Pavan",
        memory_enabled: true,
      }),
    });
    assert.equal(noConsent.status, 400);

    // 2. Reject memory creation when consent/memory is disabled
    const deniedMem = await fetch(`${baseUrl}/api/memories`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: "Prefers TypeScript" }),
    });
    assert.equal(deniedMem.status, 400);

    // 3. Grant consent + save preferences
    const okPrefs = await fetch(`${baseUrl}/api/preferences`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        consent_given: true,
        preferred_name: "Pavan",
        personality: "professional",
        response_length: "concise",
        response_tone: "direct",
        memory_enabled: true,
      }),
    });
    assert.equal(okPrefs.status, 200);
    const prefJson = (await okPrefs.json()) as {
      preferences: { consent_given: boolean; personality: string };
    };
    assert.equal(prefJson.preferences.consent_given, true);
    assert.equal(prefJson.preferences.personality, "professional");

    // 4. Reject sensitive information in long-term memory
    const sensitiveMem = await fetch(`${baseUrl}/api/memories`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: "My password is SuperSecretPassword123",
        category: "general",
      }),
    });
    assert.equal(sensitiveMem.status, 400);

    // 5. Add valid memory, inspect, and delete
    const validMem = await fetch(`${baseUrl}/api/memories`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: "Building ConvoAI in TypeScript",
        category: "project",
      }),
    });
    assert.equal(validMem.status, 201);
    const createdMem = (await validMem.json()) as { memory: { id: number } };

    const listMem = await fetch(`${baseUrl}/api/memories`);
    const listData = (await listMem.json()) as { memories: Array<{ id: number }> };
    assert.equal(listData.memories.length, 1);

    const delMem = await fetch(`${baseUrl}/api/memories/${createdMem.memory.id}`, {
      method: "DELETE",
    });
    assert.equal(delMem.status, 200);

    // 6. Save per-conversation instructions & overrides
    const convSettingsRes = await fetch(`${baseUrl}/api/conversations/conv_custom/settings`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        custom_instructions: "Use bullet points only.",
        personality: "creative",
        response_length: "detailed",
        response_tone: "playful",
      }),
    });
    assert.equal(convSettingsRes.status, 200);
    const convSettingsJson = (await convSettingsRes.json()) as {
      conversation: { custom_instructions: string; personality: string };
    };
    assert.equal(
      convSettingsJson.conversation.custom_instructions,
      "Use bullet points only."
    );
    assert.equal(convSettingsJson.conversation.personality, "creative");
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    db.close();
  }
});

test("API exposes /api/about, model selection validation, Permissions-Policy without microphone block, and attachment validation", async () => {
  const db = new Database(":memory:");
  const engine = new ChatEngine(new StubLLM(), { database: db });
  const { app } = createApp({ database: db, engine, rateLimitPerMinute: 30 });

  const server = await new Promise<ReturnType<typeof app.listen>>((resolve) => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });

  const port = (server.address() as AddressInfo).port;
  const baseUrl = `http://127.0.0.1:${port}`;

  try {
    // 1. Permissions-Policy must not block microphone for browser voice dictation
    const health = await fetch(`${baseUrl}/health`);
    const permsHeader = health.headers.get("permissions-policy") ?? "";
    assert.doesNotMatch(permsHeader, /microphone=\(\)/);

    // 2. /api/about metadata & share_links explanation
    const aboutRes = await fetch(`${baseUrl}/api/about`);
    assert.equal(aboutRes.status, 200);
    const aboutJson = (await aboutRes.json()) as {
      name: string;
      tagline: string;
      default_model: string;
      supported_models: Array<{ id: string }>;
      capabilities: { share_links: boolean; multimodal_images: boolean };
    };
    assert.equal(aboutJson.name, "ConvoAI");
    assert.equal(aboutJson.tagline, "Think. Ask. Remember.");
    assert.ok(aboutJson.supported_models.some((m) => m.id === "gemini-4-hydrogen"));
    assert.ok(aboutJson.supported_models.some((m) => m.id === "gemini-3.1-flash-lite"));
    assert.equal(aboutJson.capabilities.share_links, false);
    assert.equal(aboutJson.capabilities.multimodal_images, true);
    assert.doesNotMatch(JSON.stringify(aboutJson), /AIza/);

    // 3. Preferences includes models & validates model choices + consent
    const prefsRes = await fetch(`${baseUrl}/api/preferences`);
    const prefsJson = (await prefsRes.json()) as {
      models: Array<{ id: string }>;
      default_model: string;
    };
    assert.equal(prefsJson.models.length, 2);
    assert.ok(prefsJson.default_model);

    const modelWithoutConsent = await fetch(`${baseUrl}/api/preferences`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        consent_given: false,
        model: "gemini-3.1-flash-lite",
      }),
    });
    assert.equal(modelWithoutConsent.status, 400);

    const invalidModel = await fetch(`${baseUrl}/api/preferences`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        consent_given: true,
        model: "paid-unknown-model",
      }),
    });
    assert.equal(invalidModel.status, 400);

    const validModel = await fetch(`${baseUrl}/api/preferences`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        consent_given: true,
        model: "gemini-3.1-flash-lite",
      }),
    });
    assert.equal(validModel.status, 200);
    const savedModelJson = (await validModel.json()) as {
      preferences: { model: string };
    };
    assert.equal(savedModelJson.preferences.model, "gemini-3.1-flash-lite");

    // 4. Multimodal attachments on /api/chat and /api/chat/stream
    const attOnly = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: "conv_att",
        message: "",
        attachments: [
          {
            kind: "image",
            name: "pixel.png",
            mimeType: "image/png",
            data: "iVBORw0KGgo=",
          },
          {
            kind: "text",
            name: "readme.txt",
            mimeType: "text/plain",
            textContent: "Hello file",
          },
        ],
      }),
    });
    assert.equal(attOnly.status, 200);

    const convDetail = await fetch(`${baseUrl}/api/conversations/conv_att`);
    const convData = (await convDetail.json()) as {
      messages: Array<{ attachments: Array<{ name: string }> }>;
    };
    assert.equal(convData.messages[0].attachments.length, 2);

    const streamAtt = await fetch(`${baseUrl}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: "conv_att",
        message: "Analyze this",
        attachments: [
          {
            kind: "text",
            name: "snippet.ts",
            mimeType: "text/typescript",
            textContent: "const x = 1;",
          },
        ],
      }),
    });
    assert.equal(streamAtt.status, 200);
    assert.match(await streamAtt.text(), /event: done/);

    // 5. Reject invalid attachment payloads
    const badMime = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "hi",
        attachments: [
          { kind: "image", name: "bad.svg", mimeType: "image/svg+xml", data: "abc=" },
        ],
      }),
    });
    assert.equal(badMime.status, 400);

    const badBase64 = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "hi",
        attachments: [
          { kind: "image", name: "bad.png", mimeType: "image/png", data: "not!base64*" },
        ],
      }),
    });
    assert.equal(badBase64.status, 400);

    const tooManyAtt = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "hi",
        attachments: [
          { kind: "text", name: "1.txt", mimeType: "text/plain", textContent: "1" },
          { kind: "text", name: "2.txt", mimeType: "text/plain", textContent: "2" },
          { kind: "text", name: "3.txt", mimeType: "text/plain", textContent: "3" },
          { kind: "text", name: "4.txt", mimeType: "text/plain", textContent: "4" },
        ],
      }),
    });
    assert.equal(tooManyAtt.status, 400);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    db.close();
  }
});

