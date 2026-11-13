import test from "node:test";
import assert from "node:assert/strict";
import {
  buildClient,
  DEFAULT_MODEL,
  GeminiClient,
  type GoogleGenAIClientLike,
  mapApiError,
  UnconfiguredClient,
} from "../src/llm_client.js";
import { ChatError } from "../src/errors.js";

test("mapApiError maps Gemini HTTP error codes to user-safe ChatErrors", () => {
  const authErr = mapApiError(401, "API key not valid");
  assert.equal(authErr.status, 502);
  assert.match(authErr.userMessage, /API key/i);

  const notFound = mapApiError(404, "NOT_FOUND");
  assert.equal(notFound.status, 502);
  assert.match(notFound.userMessage, /GEMINI_MODEL/);

  const rateLimit = mapApiError(429, "RESOURCE_EXHAUSTED");
  assert.equal(rateLimit.status, 429);

  const busy = mapApiError(503, "Service unavailable");
  assert.equal(busy.status, 503);
});

test("buildClient handles missing keys and placeholder GEMINI_MODEL values safely", async () => {
  const prevKey = process.env.GEMINI_API_KEY;
  const prevModel = process.env.GEMINI_MODEL;
  const prevTimeout = process.env.REQUEST_TIMEOUT;

  try {
    process.env.GEMINI_API_KEY = "";
    const unconfigured = buildClient();
    assert.ok(unconfigured instanceof UnconfiguredClient);
    await assert.rejects(() => unconfigured.generate("sys", []), ChatError);

    process.env.GEMINI_API_KEY = "test-placeholder-only";
    process.env.GEMINI_MODEL = "your_model_name";
    process.env.REQUEST_TIMEOUT = "15";
    const configured = buildClient();
    assert.ok(configured instanceof GeminiClient);
    assert.equal(configured.getModel(), DEFAULT_MODEL);
    assert.equal(configured.getTimeoutMs(), 15000);
  } finally {
    if (prevKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = prevKey;

    if (prevModel === undefined) delete process.env.GEMINI_MODEL;
    else process.env.GEMINI_MODEL = prevModel;

    if (prevTimeout === undefined) delete process.env.REQUEST_TIMEOUT;
    else process.env.REQUEST_TIMEOUT = prevTimeout;
  }
});

test("GeminiClient succeeds for standard and streaming requests with multi-turn contents", async () => {
  let capturedParams: unknown = null;
  const mockSdk: GoogleGenAIClientLike = {
    models: {
      async generateContent(params) {
        capturedParams = params;
        return { text: "Your name is Pavan." };
      },
      async generateContentStream(params) {
        capturedParams = params;
        return {
          async *[Symbol.asyncIterator]() {
            yield { text: "Your name " };
            yield { text: "is Pavan." };
          },
        };
      },
    },
  };

  const client = new GeminiClient({
    apiKey: "AIzaFakeSecretKeyForUnitTestsOnly",
    model: "gemini-4-hydrogen",
    backoffSeconds: 0,
    client: mockSdk,
  });

  const history = [
    { role: "user" as const, text: "My name is Pavan." },
    { role: "model" as const, text: "Nice to meet you, Pavan!" },
    { role: "user" as const, text: "What is my name?" },
  ];

  const reply = await client.generate("System prompt", history);
  assert.equal(reply, "Your name is Pavan.");
  const req = capturedParams as {
    model: string;
    contents: Array<{ role: string; parts: Array<{ text: string }> }>;
    config: { systemInstruction: string };
  };
  assert.equal(req.model, "gemini-4-hydrogen");
  assert.equal(req.config.systemInstruction, "System prompt");
  assert.equal(req.contents.length, 3);
  assert.equal(req.contents[0].parts[0].text, "My name is Pavan.");

  const chunks: string[] = [];
  const streamed = await client.generateStream("System prompt", history, (c) =>
    chunks.push(c)
  );
  assert.equal(streamed, "Your name is Pavan.");
  assert.deepEqual(chunks, ["Your name ", "is Pavan."]);
});

test("GeminiClient fails fast on invalid API key, unsupported model, and rate limit without endless retries or leaking secrets", async () => {
  const secretKey = "AIzaSyFakeSecretNeverLeak123456";

  // 1. Invalid API key (403)
  let authAttempts = 0;
  const authClient = new GeminiClient({
    apiKey: secretKey,
    retries: 3,
    backoffSeconds: 0,
    client: {
      models: {
        async generateContent() {
          authAttempts++;
          throw Object.assign(
            new Error(`API key not valid: ${secretKey}`),
            { status: 403 }
          );
        },
        async generateContentStream() {
          throw new Error("unused");
        },
      },
    },
  });

  await assert.rejects(
    () => authClient.generate("sys", [{ role: "user", text: "hi" }]),
    (err: unknown) => {
      assert.ok(err instanceof ChatError);
      assert.equal(err.status, 502);
      assert.match(err.userMessage, /API key/i);
      assert.ok(!err.userMessage.includes(secretKey));
      return true;
    }
  );
  assert.equal(authAttempts, 1);

  // 2. Unavailable / unsupported model (404)
  let modelAttempts = 0;
  const modelClient = new GeminiClient({
    apiKey: secretKey,
    model: "unknown-model",
    retries: 3,
    backoffSeconds: 0,
    client: {
      models: {
        async generateContent() {
          modelAttempts++;
          throw Object.assign(new Error("models/unknown-model is NOT_FOUND"), {
            status: 404,
          });
        },
        async generateContentStream() {
          throw new Error("unused");
        },
      },
    },
  });

  await assert.rejects(
    () => modelClient.generate("sys", [{ role: "user", text: "hi" }]),
    (err: unknown) => {
      assert.ok(err instanceof ChatError);
      assert.equal(err.status, 502);
      assert.match(err.userMessage, /GEMINI_MODEL/);
      return true;
    }
  );
  assert.equal(modelAttempts, 1);

  // 3. Rate limit (429) must NOT retry endlessly
  let rateAttempts = 0;
  const rateClient = new GeminiClient({
    apiKey: secretKey,
    retries: 3,
    backoffSeconds: 0,
    client: {
      models: {
        async generateContent() {
          rateAttempts++;
          throw Object.assign(new Error("RESOURCE_EXHAUSTED"), { status: 429 });
        },
        async generateContentStream() {
          throw new Error("unused");
        },
      },
    },
  });

  await assert.rejects(
    () => rateClient.generate("sys", [{ role: "user", text: "hi" }]),
    (err: unknown) => {
      assert.ok(err instanceof ChatError);
      assert.equal(err.status, 429);
      return true;
    }
  );
  assert.equal(rateAttempts, 1);
});

test("GeminiClient retries temporary server errors (503) and enforces configurable timeouts", async () => {
  // Temporary server failure recovers on attempt 2
  let serverAttempts = 0;
  const recoveringClient = new GeminiClient({
    apiKey: "test-key",
    retries: 2,
    backoffSeconds: 0,
    client: {
      models: {
        async generateContent() {
          serverAttempts++;
          if (serverAttempts === 1) {
            throw Object.assign(new Error("Service Unavailable"), { status: 503 });
          }
          return { text: "Recovered response" };
        },
        async generateContentStream() {
          throw new Error("unused");
        },
      },
    },
  });

  const recovered = await recoveringClient.generate("sys", [
    { role: "user", text: "hi" },
  ]);
  assert.equal(recovered, "Recovered response");
  assert.equal(serverAttempts, 2);

  // Timeout triggers 504 after configured timeoutSeconds
  let timeoutAttempts = 0;
  const slowClient = new GeminiClient({
    apiKey: "test-key",
    timeoutSeconds: 0.1,
    retries: 1,
    backoffSeconds: 0,
    client: {
      models: {
        async generateContent({ config }) {
          timeoutAttempts++;
          return new Promise((_resolve, reject) => {
            config?.abortSignal?.addEventListener("abort", () => {
              reject(new Error("Aborted due to timeout"));
            });
          });
        },
        async generateContentStream() {
          throw new Error("unused");
        },
      },
    },
  });

  await assert.rejects(
    () => slowClient.generate("sys", [{ role: "user", text: "hi" }]),
    (err: unknown) => {
      assert.ok(err instanceof ChatError);
      assert.equal(err.status, 504);
      assert.match(err.userMessage, /too long/i);
      return true;
    }
  );
  assert.equal(timeoutAttempts, 2);
});

test("GeminiClient handles empty, safety-blocked, and malformed responses cleanly", async () => {
  const mkClient = (response: unknown) =>
    new GeminiClient({
      apiKey: "test-key",
      retries: 1,
      backoffSeconds: 0,
      client: {
        models: {
          async generateContent() {
            return response as Record<string, unknown>;
          },
          async generateContentStream() {
            throw new Error("unused");
          },
        },
      },
    });

  // Empty response
  await assert.rejects(
    () => mkClient({ text: "   " }).generate("sys", [{ role: "user", text: "hi" }]),
    (err: unknown) => {
      assert.ok(err instanceof ChatError);
      assert.equal(err.status, 502);
      assert.match(err.userMessage, /empty/i);
      return true;
    }
  );

  // Blocked safety response
  await assert.rejects(
    () =>
      mkClient({
        text: "",
        promptFeedback: { blockReason: "SAFETY" },
      }).generate("sys", [{ role: "user", text: "hi" }]),
    (err: unknown) => {
      assert.ok(err instanceof ChatError);
      assert.equal(err.status, 422);
      assert.match(err.userMessage, /safety/i);
      return true;
    }
  );

  // Malformed provider response
  await assert.rejects(
    () => mkClient(null).generate("sys", [{ role: "user", text: "hi" }]),
    (err: unknown) => {
      assert.ok(err instanceof ChatError);
      assert.equal(err.status, 502);
      assert.match(err.userMessage, /malformed/i);
      return true;
    }
  );
});

test("GeminiClient encodes multimodal image/PDF inlineData, text document parts, and honors per-request model overrides", async () => {
  const { buildContentParts } = await import("../src/llm_client.js");
  const seenParams: Array<{
    model: string;
    contents: Array<{
      role: string;
      parts: Array<{ text?: string; inlineData?: { mimeType: string; data: string } }>;
    }>;
  }> = [];

  const mockSdk: GoogleGenAIClientLike = {
    models: {
      async generateContent(params) {
        seenParams.push(params as (typeof seenParams)[number]);
        return { text: "Analyzed multimodal input." };
      },
      async generateContentStream(params) {
        seenParams.push(params as (typeof seenParams)[number]);
        return {
          async *[Symbol.asyncIterator]() {
            yield { text: "Streamed multimodal " };
            yield { text: "analysis." };
          },
        };
      },
    },
  };

  const client = new GeminiClient({
    apiKey: "test-key",
    model: "gemini-4-hydrogen",
    backoffSeconds: 0,
    client: mockSdk,
  });

  const history = [
    {
      role: "user" as const,
      text: "Check these files.",
      attachments: [
        {
          kind: "image" as const,
          name: "chart.png",
          mimeType: "image/png",
          data: "iVBORw0KGgo=",
        },
        {
          kind: "pdf" as const,
          name: "report.pdf",
          mimeType: "application/pdf",
          data: "JVBERi0xLjQ=",
        },
        {
          kind: "text" as const,
          name: "config.json",
          mimeType: "application/json",
          textContent: '{"env":"prod"}',
        },
      ],
    },
  ];

  const built = buildContentParts(history[0]);
  assert.equal(built.length, 3);
  assert.ok("text" in built[0] && built[0].text.includes("Attached Document: config.json"));
  assert.ok("text" in built[0] && built[0].text.includes('{"env":"prod"}'));
  assert.deepEqual(built[1], {
    inlineData: { mimeType: "image/png", data: "iVBORw0KGgo=" },
  });
  assert.deepEqual(built[2], {
    inlineData: { mimeType: "application/pdf", data: "JVBERi0xLjQ=" },
  });

  const reply = await client.generate("sys", history, {
    model: "gemini-3.1-flash-lite",
  });
  assert.equal(reply, "Analyzed multimodal input.");
  assert.equal(seenParams[0].model, "gemini-3.1-flash-lite");
  assert.equal(seenParams[0].contents[0].parts.length, 3);

  const chunks: string[] = [];
  const streamed = await client.generateStream(
    "sys",
    history,
    (c) => chunks.push(c),
    { model: "gemini-3.1-flash-lite" }
  );
  assert.equal(streamed, "Streamed multimodal analysis.");
  assert.equal(seenParams[1].model, "gemini-3.1-flash-lite");
});

