import test from "node:test";
import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { loadConfig } from "../src/config.js";
import { Database, type HistoryMessage } from "../src/database.js";
import { ChatEngine } from "../src/chat_engine.js";
import { redactSecrets, type LLMClient } from "../src/llm_client.js";
import { createApp } from "../src/server.js";

class BrokenSecretMockLLM implements LLMClient {
  async generate(_system: string, messages: HistoryMessage[]): Promise<string> {
    const last = messages[messages.length - 1]?.text ?? "";
    if (last === "crash-with-secret") {
      throw new Error(
        "Unexpected transport failure key=AIzaSySuperSecretKey1234567890123 Bearer tok_abcdef1234567890"
      );
    }
    return `Safe reply: ${last}`;
  }
}

test("loadConfig separates development and production security defaults", () => {
  const devCfg = loadConfig({
    NODE_ENV: "development",
    CORS_ORIGINS: "*,http://localhost:5173",
  });
  assert.equal(devCfg.nodeEnv, "development");
  assert.equal(devCfg.isProduction, false);
  assert.equal(devCfg.trustProxy, false);
  assert.deepEqual(devCfg.corsOrigins, ["*", "http://localhost:5173"]);

  const prodCfg = loadConfig({
    NODE_ENV: "production",
    CORS_ORIGINS: "*,https://convoai.example.com",
    DATABASE_PATH: "/tmp/convoai-prod-test.db",
    PORT: "8080",
  });
  assert.equal(prodCfg.nodeEnv, "production");
  assert.equal(prodCfg.isProduction, true);
  assert.equal(prodCfg.trustProxy, 1);
  assert.equal(prodCfg.port, 8080);
  // Wildcard "*" must be stripped in production
  assert.deepEqual(prodCfg.corsOrigins, ["https://convoai.example.com"]);
});

test("redactSecrets strips Gemini API keys, Bearer tokens, and query keys", () => {
  const raw =
    "Failed https://generativelanguage.googleapis.com/v1?key=AIzaSyFakeKey1234567890123456 with Bearer secret-token-999999";
  const clean = redactSecrets(raw, "explicit-secret-value");
  assert.doesNotMatch(clean, /AIzaSyFakeKey/);
  assert.doesNotMatch(clean, /secret-token-999999/);
  assert.match(clean, /\[REDACTED/);
});

test("Database LIKE search escapes wildcards and backslashes safely", () => {
  const db = new Database(":memory:");
  try {
    db.createConversation("c1");
    db.addMessage("c1", "user", "Literal path C:\\temp\\notes_100%");
    db.createConversation("c2");
    db.addMessage("c2", "user", "Regular chat message");

    assert.equal(db.getConversations("\\temp\\").length, 1);
    assert.equal(db.getConversations("100%").length, 1);
    assert.equal(db.getConversations("notes_100").length, 1);
    // Searching "%" must NOT match unrelated rows as a wildcard
    assert.equal(db.getConversations("%").length, 1);
  } finally {
    db.close();
  }
});

test("Server enforces production headers, JSON size limits, proxy rate limiting, secret-free errors, and honest auth metadata", async () => {
  const db = new Database(":memory:");
  const engine = new ChatEngine(new BrokenSecretMockLLM(), { database: db });
  const { app } = createApp({
    database: db,
    engine,
    rateLimitPerMinute: 2,
    config: {
      nodeEnv: "production",
      isProduction: true,
      trustProxy: 1,
      corsOrigins: ["https://allowed.example.com"],
    },
  });

  const server = await new Promise<ReturnType<typeof app.listen>>((resolve) => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });

  const port = (server.address() as AddressInfo).port;
  const baseUrl = `http://127.0.0.1:${port}`;

  try {
    // 1. Security headers + X-Powered-By disabled + HSTS on HTTPS forwarded requests
    const aboutRes = await fetch(`${baseUrl}/api/about`, {
      headers: {
        Origin: "https://evil.example.com",
        "X-Forwarded-Proto": "https",
      },
    });
    assert.equal(aboutRes.status, 200);
    assert.equal(aboutRes.headers.get("x-powered-by"), null);
    assert.equal(aboutRes.headers.get("x-content-type-options"), "nosniff");
    assert.match(
      aboutRes.headers.get("content-security-policy") ?? "",
      /object-src 'none'/
    );
    assert.match(
      aboutRes.headers.get("strict-transport-security") ?? "",
      /max-age=31536000/
    );
    // Unlisted origin gets no ACAO header
    assert.equal(aboutRes.headers.get("access-control-allow-origin"), null);

    const aboutBody = (await aboutRes.json()) as {
      environment: string;
      capabilities: { authentication: boolean; per_user_isolation: boolean; share_links: boolean };
    };
    assert.equal(aboutBody.environment, "production");
    assert.equal(aboutBody.capabilities.authentication, false);
    assert.equal(aboutBody.capabilities.per_user_isolation, false);
    assert.equal(aboutBody.capabilities.share_links, false);

    // Allowed CORS origin succeeds with PUT included
    const corsRes = await fetch(`${baseUrl}/api/preferences`, {
      method: "OPTIONS",
      headers: { Origin: "https://allowed.example.com" },
    });
    assert.equal(corsRes.status, 204);
    assert.equal(
      corsRes.headers.get("access-control-allow-origin"),
      "https://allowed.example.com"
    );
    assert.match(corsRes.headers.get("access-control-allow-methods") ?? "", /\bPUT\b/);

    // 2. Non-chat endpoints reject payloads > 64 KB
    const oversizedPrefs = await fetch(`${baseUrl}/api/preferences`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        consent_given: true,
        custom_system_prompt: "a".repeat(80_000),
      }),
    });
    assert.equal(oversizedPrefs.status, 413);

    // 3. Search query > 200 chars rejected
    const longSearch = await fetch(`${baseUrl}/api/conversations?q=${"x".repeat(205)}`);
    assert.equal(longSearch.status, 400);

    // 4. Unknown /api route returns JSON 404 (not Express HTML)
    const unknownApi = await fetch(`${baseUrl}/api/does-not-exist`);
    assert.equal(unknownApi.status, 404);
    const unknownJson = (await unknownApi.json()) as { error: string };
    assert.match(unknownJson.error, /not found/i);

    // 5. Internal errors never expose stack traces or secrets to HTTP callers
    const crashRes = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Forwarded-For": "203.0.113.10",
      },
      body: JSON.stringify({ message: "crash-with-secret" }),
    });
    assert.equal(crashRes.status, 500);
    const crashText = await crashRes.text();
    assert.doesNotMatch(crashText, /AIzaSySuperSecret/);
    assert.doesNotMatch(crashText, /stack/i);

    // 6. Rate limiter respects X-Forwarded-For under trustProxy and returns Retry-After on 429
    const r2 = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Forwarded-For": "203.0.113.10",
      },
      body: JSON.stringify({ message: "hello" }),
    });
    assert.equal(r2.status, 200);

    const r3RateLimited = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Forwarded-For": "203.0.113.10",
      },
      body: JSON.stringify({ message: "blocked" }),
    });
    assert.equal(r3RateLimited.status, 429);
    assert.equal(r3RateLimited.headers.get("retry-after"), "60");

    // Different client IP is NOT rate-limited
    const otherClient = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Forwarded-For": "203.0.113.99",
      },
      body: JSON.stringify({ message: "different client" }),
    });
    assert.equal(otherClient.status, 200);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    db.close();
  }
});
