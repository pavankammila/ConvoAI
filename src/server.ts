/**
 * ConvoAI - Express Backend
 *
 * Serves static/index.html, brand assets, SQLite-backed conversation APIs,
 * and standard + SSE streaming Gemini chat endpoints.
 */

import "dotenv/config";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express, { type NextFunction, type Request, type Response } from "express";

import {
  ChatEngine,
  MAX_MESSAGE_CHARS,
  PERSONALITY_PROMPTS,
} from "./chat_engine.js";
import {
  Database,
  type MemoryCategory,
  type MessageAttachment,
  type PersonalityId,
  type ResponseLength,
  type ResponseTone,
} from "./database.js";
import { loadConfig, type AppConfig } from "./config.js";
import { ChatError } from "./errors.js";
import { buildClient, redactSecrets, SUPPORTED_MODELS } from "./llm_client.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, "..");
const STATIC_DIR = path.join(ROOT_DIR, "static");
const ASSETS_DIR = path.join(ROOT_DIR, "assets");

const CONVERSATION_ID_RE = /^[a-zA-Z0-9_-]{1,64}$/;
const VALID_PERSONALITIES = new Set<PersonalityId>([
  "friendly",
  "professional",
  "coding",
  "study",
  "creative",
  "custom",
]);
const VALID_LENGTHS = new Set<ResponseLength>(["concise", "balanced", "detailed"]);
const VALID_TONES = new Set<ResponseTone>(["warm", "neutral", "direct", "playful"]);
const VALID_MEMORY_CATEGORIES = new Set<MemoryCategory>([
  "preference",
  "goal",
  "background",
  "project",
  "general",
]);
const VALID_MODEL_IDS = new Set(SUPPORTED_MODELS.map((m) => m.id));
const ALLOWED_IMAGE_MIMES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
]);
const ALLOWED_TEXT_MIMES = new Set([
  "text/plain",
  "text/markdown",
  "text/csv",
  "text/html",
  "text/css",
  "text/javascript",
  "text/typescript",
  "text/x-python",
  "text/x-typescript",
  "text/xml",
  "text/yaml",
  "application/json",
  "application/javascript",
  "application/xml",
  "application/x-yaml",
  "application/sql",
]);
const MAX_ATTACHMENTS = 3;
const MAX_BINARY_BASE64_CHARS = 5_600_000;
const MAX_TEXT_ATTACHMENT_CHARS = 250_000;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateConversationId(raw: unknown): string | null {
  if (raw === undefined || raw === null || raw === "") {
    return null;
  }
  if (typeof raw !== "string" || !CONVERSATION_ID_RE.test(raw)) {
    throw new ChatError(
      "Invalid conversation ID format. Use 1-64 alphanumeric, hyphen, or underscore characters.",
      400
    );
  }
  return raw;
}

function validateAttachments(raw: unknown): MessageAttachment[] {
  if (raw === undefined || raw === null) {
    return [];
  }
  if (!Array.isArray(raw)) {
    throw new ChatError("Attachments must be a list.", 400);
  }
  if (raw.length > MAX_ATTACHMENTS) {
    throw new ChatError(`You can attach up to ${MAX_ATTACHMENTS} files per message.`, 400);
  }

  const clean: MessageAttachment[] = [];
  for (const item of raw) {
    if (!isPlainObject(item)) {
      throw new ChatError("Invalid attachment object.", 400);
    }

    const name =
      typeof item.name === "string"
        ? item.name.trim().replace(/[\r\n\0]/g, "").slice(0, 120)
        : "";
    if (!name) {
      throw new ChatError("Each attachment must include a valid file name.", 400);
    }

    const kind = item.kind;
    const mimeType = typeof item.mimeType === "string" ? item.mimeType.trim().toLowerCase() : "";

    if (kind === "image") {
      if (!ALLOWED_IMAGE_MIMES.has(mimeType)) {
        throw new ChatError(
          `Unsupported image format (${mimeType || "unknown"}). Use PNG, JPEG, WEBP, or GIF.`,
          400
        );
      }
      if (
        typeof item.data !== "string" ||
        item.data.length === 0 ||
        item.data.length > MAX_BINARY_BASE64_CHARS ||
        item.data.length % 4 !== 0 ||
        !/^[A-Za-z0-9+/]+={0,2}$/.test(item.data)
      ) {
        throw new ChatError(`Image "${name}" is invalid or exceeds the 4 MB limit.`, 400);
      }
      clean.push({ kind: "image", name, mimeType, data: item.data });
    } else if (kind === "pdf") {
      if (mimeType !== "application/pdf") {
        throw new ChatError("PDF attachments must use application/pdf.", 400);
      }
      if (
        typeof item.data !== "string" ||
        item.data.length === 0 ||
        item.data.length > MAX_BINARY_BASE64_CHARS ||
        item.data.length % 4 !== 0 ||
        !/^[A-Za-z0-9+/]+={0,2}$/.test(item.data)
      ) {
        throw new ChatError(`PDF "${name}" is invalid or exceeds the 4 MB limit.`, 400);
      }
      clean.push({ kind: "pdf", name, mimeType, data: item.data });
    } else if (kind === "text") {
      const normalizedMime =
        mimeType && (ALLOWED_TEXT_MIMES.has(mimeType) || mimeType.startsWith("text/"))
          ? mimeType
          : "text/plain";
      if (typeof item.textContent !== "string") {
        throw new ChatError(`Document "${name}" content must be text.`, 400);
      }
      if (item.textContent.length > MAX_TEXT_ATTACHMENT_CHARS) {
        throw new ChatError(
          `Document "${name}" exceeds the ${Math.round(MAX_TEXT_ATTACHMENT_CHARS / 1000)} KB text limit.`,
          400
        );
      }
      clean.push({
        kind: "text",
        name,
        mimeType: normalizedMime,
        textContent: item.textContent,
      });
    } else {
      throw new ChatError("Unsupported attachment type.", 400);
    }
  }

  return clean;
}

function createRateLimiter(maxPerMinute: number) {
  const hits = new Map<string, number[]>();

  return (req: Request, res: Response, next: NextFunction) => {
    if (maxPerMinute <= 0) {
      next();
      return;
    }

    const now = Date.now();
    const windowStart = now - 60_000;
    const key = req.ip || req.socket.remoteAddress || "unknown";
    const recent = (hits.get(key) ?? []).filter((ts) => ts > windowStart);

    if (recent.length >= maxPerMinute) {
      res.setHeader("Retry-After", "60");
      res.status(429).json({
        error: "Too many messages sent. Please wait a moment and try again.",
      });
      return;
    }

    recent.push(now);
    hits.set(key, recent);

    if (hits.size > 1000) {
      for (const [ip, timestamps] of hits) {
        if (timestamps.every((ts) => ts <= windowStart)) {
          hits.delete(ip);
        }
      }
    }

    next();
  };
}

function writeSseEvent(res: Response, event: string, data: unknown): void {
  if (res.writableEnded) return;
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

export function createApp(options?: {
  engine?: ChatEngine;
  database?: Database;
  rateLimitPerMinute?: number;
  config?: Partial<AppConfig>;
}) {
  const baseConfig = loadConfig();
  const config: AppConfig = {
    ...baseConfig,
    ...(options?.config ?? {}),
    rateLimitPerMinute:
      options?.rateLimitPerMinute ??
      options?.config?.rateLimitPerMinute ??
      baseConfig.rateLimitPerMinute,
  };

  const app = express();
  app.disable("x-powered-by");
  if (config.trustProxy !== false) {
    app.set("trust proxy", config.trustProxy);
  }

  const database = options?.database ?? new Database(config.databasePath);
  const llm = buildClient();
  const engine =
    options?.engine ??
    new ChatEngine(llm, {
      database,
    });

  const chatLimiter = createRateLimiter(config.rateLimitPerMinute);
  const standardJsonParser = express.json({ limit: "64kb" });
  const chatJsonParser = express.json({ limit: "16mb" });

  // Security headers (iframe-compatible for preview while enforcing strict CSP/HSTS)
  app.use((req: Request, res: Response, next: NextFunction) => {
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
    res.setHeader("Permissions-Policy", "camera=(), geolocation=()");
    res.setHeader(
      "Content-Security-Policy",
      [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline'",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com data:",
        "img-src 'self' data: blob:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
      ].join("; ")
    );
    if (
      config.isProduction &&
      (req.secure || req.headers["x-forwarded-proto"] === "https")
    ) {
      res.setHeader(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains"
      );
    }
    next();
  });

  // Strict 64 KB default JSON limit; chat/stream routes opt into 16 MB for multimodal uploads
  app.use((req: Request, res: Response, next: NextFunction) => {
    if (req.path === "/api/chat" || req.path === "/api/chat/stream") {
      chatJsonParser(req, res, next);
      return;
    }
    standardJsonParser(req, res, next);
  });

  // Handle malformed or oversized JSON bodies cleanly
  app.use((err: unknown, _req: Request, res: Response, next: NextFunction) => {
    if (err && typeof err === "object" && "type" in err && (err as { type?: string }).type === "entity.too.large") {
      res.status(413).json({
        error: "Request body or attached files exceed the allowed size limit.",
      });
      return;
    }
    if (err instanceof SyntaxError && "body" in err) {
      res.status(400).json({
        error: "Invalid JSON request body.",
      });
      return;
    }
    next(err);
  });

  // CORS (same-origin by default; production ignores wildcard "*")
  const origins = config.corsOrigins;
  if (origins.length > 0) {
    app.use((req: Request, res: Response, next: NextFunction) => {
      const origin = req.headers.origin;
      if (origin && (origins.includes("*") || origins.includes(origin))) {
        res.setHeader("Access-Control-Allow-Origin", origin);
        res.setHeader("Vary", "Origin");
        res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE");
        res.setHeader("Access-Control-Allow-Headers", "Content-Type");
      }
      if (req.method === "OPTIONS") {
        res.status(204).end();
        return;
      }
      next();
    });
  }

  // Static brand assets
  app.use("/assets", express.static(ASSETS_DIR));
  app.use("/static/assets", express.static(ASSETS_DIR));

  // Frontend
  app.get("/", (_req: Request, res: Response) => {
    res.sendFile(path.join(STATIC_DIR, "index.html"));
  });

  // Basic endpoints
  app.get("/api", (_req: Request, res: Response) => {
    res.json({
      message: "Welcome to ConvoAI!",
      status: "Backend is running",
      docs: "/api",
    });
  });

  app.get("/health", (_req: Request, res: Response) => {
    res.json({ status: "healthy" });
  });

  app.get("/api/about", (_req: Request, res: Response) => {
    res.json({
      name: "ConvoAI",
      tagline: "Think. Ask. Remember.",
      environment: config.nodeEnv,
      default_model: engine.getDefaultModelName(),
      supported_models: SUPPORTED_MODELS,
      storage: "SQLite (local persistent database)",
      capabilities: {
        streaming: true,
        multimodal_images: true,
        document_analysis: true,
        long_term_memory: true,
        authentication: false,
        per_user_isolation: false,
        share_links: false,
        share_links_reason:
          "Conversation sharing via public links and multi-user private isolation require authentication and per-user access controls.",
      },
    });
  });

  // Standard chat endpoint
  app.post(
    "/api/chat",
    chatLimiter,
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        if (!isPlainObject(req.body)) {
          throw new ChatError(
            `Invalid request: message must contain 1-${MAX_MESSAGE_CHARS} characters.`,
            400
          );
        }

        const rawMessage = req.body.message ?? "";
        if (typeof rawMessage !== "string") {
          throw new ChatError(
            `Invalid request: message must contain 1-${MAX_MESSAGE_CHARS} characters.`,
            400
          );
        }

        const trimmed = rawMessage.trim();
        const attachments = validateAttachments(req.body.attachments);
        if ((!trimmed && attachments.length === 0) || trimmed.length > MAX_MESSAGE_CHARS) {
          throw new ChatError(
            `Invalid request: message must contain 1-${MAX_MESSAGE_CHARS} characters or valid attachments.`,
            400
          );
        }

        const conversationId = validateConversationId(req.body.conversation_id);
        const response = await engine.send(conversationId, trimmed, attachments);
        res.json(response);
      } catch (error) {
        next(error);
      }
    }
  );

  // Streaming SSE chat endpoint
  app.post(
    "/api/chat/stream",
    chatLimiter,
    async (req: Request, res: Response, next: NextFunction) => {
      let trimmed = "";
      let attachments: MessageAttachment[] = [];
      let conversationId: string | null = null;

      try {
        if (!isPlainObject(req.body)) {
          throw new ChatError(
            `Invalid request: message must contain 1-${MAX_MESSAGE_CHARS} characters.`,
            400
          );
        }

        const rawMessage = req.body.message ?? "";
        if (typeof rawMessage !== "string") {
          throw new ChatError(
            `Invalid request: message must contain 1-${MAX_MESSAGE_CHARS} characters.`,
            400
          );
        }

        trimmed = rawMessage.trim();
        attachments = validateAttachments(req.body.attachments);
        if ((!trimmed && attachments.length === 0) || trimmed.length > MAX_MESSAGE_CHARS) {
          throw new ChatError(
            `Invalid request: message must contain 1-${MAX_MESSAGE_CHARS} characters or valid attachments.`,
            400
          );
        }

        conversationId = validateConversationId(req.body.conversation_id);
      } catch (err) {
        next(err);
        return;
      }

      res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
      res.setHeader("Cache-Control", "no-cache, no-transform");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      res.flushHeaders?.();

      const controller = new AbortController();
      const onClose = () => {
        if (!res.writableEnded) {
          controller.abort();
        }
      };
      res.on("close", onClose);

      try {
        const result = await engine.sendStream(
          conversationId,
          trimmed,
          {
            signal: controller.signal,
            onMeta: (meta) => writeSseEvent(res, "meta", meta),
            onChunk: (chunk) => writeSseEvent(res, "chunk", { text: chunk }),
          },
          attachments
        );

        writeSseEvent(res, "done", result);
      } catch (error) {
        if (!controller.signal.aborted) {
          const message =
            error instanceof ChatError
              ? error.userMessage
              : "Unexpected server error. Please try again.";
          writeSseEvent(res, "error", { error: message });
        }
      } finally {
        res.off("close", onClose);
        if (!res.writableEnded) {
          res.end();
        }
      }
    }
  );

  // Regenerate last assistant response via SSE
  app.post(
    "/api/chat/regenerate",
    chatLimiter,
    async (req: Request, res: Response, next: NextFunction) => {
      let conversationId: string | null;
      try {
        if (!isPlainObject(req.body)) {
          throw new ChatError("Invalid request body.", 400);
        }
        conversationId = validateConversationId(req.body.conversation_id);
        if (!conversationId) {
          throw new ChatError("Conversation ID is required.", 400);
        }
      } catch (err) {
        next(err);
        return;
      }

      res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
      res.setHeader("Cache-Control", "no-cache, no-transform");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      res.flushHeaders?.();

      const controller = new AbortController();
      const onClose = () => {
        if (!res.writableEnded) {
          controller.abort();
        }
      };
      res.on("close", onClose);

      try {
        const result = await engine.regenerateStream(conversationId, {
          signal: controller.signal,
          onMeta: (meta) => writeSseEvent(res, "meta", meta),
          onChunk: (chunk) => writeSseEvent(res, "chunk", { text: chunk }),
        });

        writeSseEvent(res, "done", result);
      } catch (error) {
        if (!controller.signal.aborted) {
          const message =
            error instanceof ChatError
              ? error.userMessage
              : "Unexpected server error. Please try again.";
          writeSseEvent(res, "error", { error: message });
        }
      } finally {
        res.off("close", onClose);
        if (!res.writableEnded) {
          res.end();
        }
      }
    }
  );

  // Reset endpoint
  app.post("/api/reset", (req: Request, res: Response, next: NextFunction) => {
    try {
      if (!isPlainObject(req.body)) {
        throw new ChatError("Invalid request body.", 400);
      }
      const conversationId = validateConversationId(req.body.conversation_id);
      if (conversationId) {
        engine.reset(conversationId);
      }
      res.json({ status: "reset" });
    } catch (error) {
      next(error);
    }
  });

  // List / search conversations
  app.get("/api/conversations", (req: Request, res: Response, next: NextFunction) => {
    try {
      const rawQuery = req.query.q;
      if (rawQuery !== undefined && typeof rawQuery !== "string") {
        throw new ChatError("Search query must be text.", 400);
      }
      const query = typeof rawQuery === "string" ? rawQuery.trim() : undefined;
      if (query && query.length > 200) {
        throw new ChatError("Search query must be 200 characters or fewer.", 400);
      }
      const conversations = database.getConversations(query);
      res.json({
        conversations,
        total: conversations.length,
      });
    } catch (error) {
      next(error);
    }
  });

  // Get conversation details + full transcript
  app.get(
    "/api/conversations/:conversation_id",
    (req: Request, res: Response, next: NextFunction) => {
      try {
        const conversationId = validateConversationId(req.params.conversation_id);
        if (!conversationId) {
          throw new ChatError("Conversation not found.", 404);
        }

        const conversation = database.getConversation(conversationId);
        if (!conversation) {
          res.status(404).json({ error: "Conversation not found." });
          return;
        }

        const messages = database.getMessages(conversationId);
        const contextMessages = database.getContextMessages(conversationId);

        res.json({
          conversation,
          messages,
          messages_in_context: contextMessages.length,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  // Rename conversation
  app.patch(
    "/api/conversations/:conversation_id",
    (req: Request, res: Response, next: NextFunction) => {
      try {
        const conversationId = validateConversationId(req.params.conversation_id);
        if (!conversationId) {
          throw new ChatError("Conversation not found.", 404);
        }
        if (!isPlainObject(req.body) || typeof req.body.title !== "string") {
          throw new ChatError("Title must be 1-80 characters.", 400);
        }

        const title = req.body.title.trim();
        if (!title || title.length > 80) {
          throw new ChatError("Title must be 1-80 characters.", 400);
        }

        const updated = database.renameConversation(conversationId, title);
        if (!updated) {
          res.status(404).json({ error: "Conversation not found." });
          return;
        }

        res.json({
          status: "renamed",
          conversation_id: conversationId,
          title,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  // Clear conversation memory summary
  app.delete(
    "/api/conversations/:conversation_id/summary",
    (req: Request, res: Response, next: NextFunction) => {
      try {
        const conversationId = validateConversationId(req.params.conversation_id);
        if (!conversationId) {
          throw new ChatError("Conversation not found.", 404);
          return;
        }

        const cleared = engine.clearSummary(conversationId);
        if (!cleared) {
          res.status(404).json({ error: "Conversation not found." });
          return;
        }

        res.json({
          status: "summary_cleared",
          conversation_id: conversationId,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  // Update per-conversation instructions & overrides
  app.patch(
    "/api/conversations/:conversation_id/settings",
    (req: Request, res: Response, next: NextFunction) => {
      try {
        const conversationId = validateConversationId(req.params.conversation_id);
        if (!conversationId) {
          throw new ChatError("Invalid conversation ID.", 400);
        }
        if (!isPlainObject(req.body)) {
          throw new ChatError("Invalid request body.", 400);
        }

        const patch: {
          custom_instructions?: string;
          personality?: PersonalityId | "";
          response_length?: ResponseLength | "";
          response_tone?: ResponseTone | "";
        } = {};

        if (req.body.custom_instructions !== undefined) {
          if (typeof req.body.custom_instructions !== "string") {
            throw new ChatError("Conversation instructions must be text.", 400);
          }
          patch.custom_instructions = req.body.custom_instructions.trim().slice(0, 1000);
        }

        if (req.body.personality !== undefined) {
          if (
            req.body.personality !== "" &&
            (typeof req.body.personality !== "string" ||
              !VALID_PERSONALITIES.has(req.body.personality as PersonalityId))
          ) {
            throw new ChatError("Invalid personality selection.", 400);
          }
          patch.personality = req.body.personality as PersonalityId | "";
        }

        if (req.body.response_length !== undefined) {
          if (
            req.body.response_length !== "" &&
            (typeof req.body.response_length !== "string" ||
              !VALID_LENGTHS.has(req.body.response_length as ResponseLength))
          ) {
            throw new ChatError("Invalid response length.", 400);
          }
          patch.response_length = req.body.response_length as ResponseLength | "";
        }

        if (req.body.response_tone !== undefined) {
          if (
            req.body.response_tone !== "" &&
            (typeof req.body.response_tone !== "string" ||
              !VALID_TONES.has(req.body.response_tone as ResponseTone))
          ) {
            throw new ChatError("Invalid response tone.", 400);
          }
          patch.response_tone = req.body.response_tone as ResponseTone | "";
        }

        if (!database.getConversation(conversationId)) {
          database.createConversation(conversationId);
        }

        const updated = database.updateConversationSettings(conversationId, patch);
        if (!updated) {
          res.status(404).json({ error: "Conversation not found." });
          return;
        }

        engine.syncConversationSettings(conversationId, {
          custom_instructions: updated.custom_instructions,
          personality: updated.personality,
          response_length: updated.response_length,
          response_tone: updated.response_tone,
        });

        res.json({ conversation: updated });
      } catch (error) {
        next(error);
      }
    }
  );

  // User preferences (consent-backed persistence)
  app.get("/api/preferences", (_req: Request, res: Response) => {
    res.json({
      preferences: database.getPreferences(),
      presets: PERSONALITY_PROMPTS,
      models: SUPPORTED_MODELS,
      default_model: engine.getDefaultModelName(),
    });
  });

  app.put(
    "/api/preferences",
    (req: Request, res: Response, next: NextFunction) => {
      try {
        if (!isPlainObject(req.body)) {
          throw new ChatError("Invalid request body.", 400);
        }

        const consentGiven =
          typeof req.body.consent_given === "boolean"
            ? req.body.consent_given
            : database.getPreferences().consent_given;

        if (
          !consentGiven &&
          (req.body.preferred_name ||
            req.body.custom_system_prompt ||
            req.body.model ||
            req.body.memory_enabled === true)
        ) {
          throw new ChatError(
            "User consent is required before saving persistent preferences or memory.",
            400
          );
        }

        const patch: {
          consent_given: boolean;
          preferred_name?: string;
          personality?: PersonalityId;
          custom_system_prompt?: string;
          response_length?: ResponseLength;
          response_tone?: ResponseTone;
          memory_enabled?: boolean;
          model?: string;
        } = { consent_given: consentGiven };

        if (req.body.preferred_name !== undefined) {
          if (typeof req.body.preferred_name !== "string") {
            throw new ChatError("Preferred name must be text.", 400);
          }
          patch.preferred_name = req.body.preferred_name;
        }

        if (req.body.personality !== undefined) {
          if (
            typeof req.body.personality !== "string" ||
            !VALID_PERSONALITIES.has(req.body.personality as PersonalityId)
          ) {
            throw new ChatError("Invalid personality selection.", 400);
          }
          patch.personality = req.body.personality as PersonalityId;
        }

        if (req.body.custom_system_prompt !== undefined) {
          if (typeof req.body.custom_system_prompt !== "string") {
            throw new ChatError("Custom personality prompt must be text.", 400);
          }
          patch.custom_system_prompt = req.body.custom_system_prompt;
        }

        if (req.body.response_length !== undefined) {
          if (
            typeof req.body.response_length !== "string" ||
            !VALID_LENGTHS.has(req.body.response_length as ResponseLength)
          ) {
            throw new ChatError("Invalid response length.", 400);
          }
          patch.response_length = req.body.response_length as ResponseLength;
        }

        if (req.body.response_tone !== undefined) {
          if (
            typeof req.body.response_tone !== "string" ||
            !VALID_TONES.has(req.body.response_tone as ResponseTone)
          ) {
            throw new ChatError("Invalid response tone.", 400);
          }
          patch.response_tone = req.body.response_tone as ResponseTone;
        }

        if (req.body.memory_enabled !== undefined) {
          if (typeof req.body.memory_enabled !== "boolean") {
            throw new ChatError("memory_enabled must be a boolean.", 400);
          }
          patch.memory_enabled = req.body.memory_enabled;
        }

        if (req.body.model !== undefined) {
          if (typeof req.body.model !== "string") {
            throw new ChatError("Model selection must be text.", 400);
          }
          const m = req.body.model.trim();
          if (m && !VALID_MODEL_IDS.has(m)) {
            throw new ChatError("Unsupported model selection.", 400);
          }
          patch.model = m;
        }

        const saved = database.savePreferences(patch);
        res.json({ preferences: saved });
      } catch (error) {
        next(error);
      }
    }
  );

  app.delete("/api/preferences", (_req: Request, res: Response) => {
    const preferences = database.resetPreferences();
    res.json({ preferences });
  });

  // Explicit long-term user memories
  app.get("/api/memories", (_req: Request, res: Response) => {
    const prefs = database.getPreferences();
    res.json({
      consent_given: prefs.consent_given,
      memory_enabled: prefs.memory_enabled,
      memories: database.getMemories(),
    });
  });

  app.post(
    "/api/memories",
    (req: Request, res: Response, next: NextFunction) => {
      try {
        const prefs = database.getPreferences();
        if (!prefs.consent_given || !prefs.memory_enabled) {
          throw new ChatError(
            "Enable preference consent and long-term memory before saving memories.",
            400
          );
        }

        if (!isPlainObject(req.body) || typeof req.body.content !== "string") {
          throw new ChatError("Memory content is required.", 400);
        }

        const rawCategory = req.body.category ?? "general";
        if (
          typeof rawCategory !== "string" ||
          !VALID_MEMORY_CATEGORIES.has(rawCategory as MemoryCategory)
        ) {
          throw new ChatError("Invalid memory category.", 400);
        }

        const sourceConvId = validateConversationId(req.body.source_conversation_id);
        let memory;
        try {
          memory = database.addMemory(
            req.body.content,
            rawCategory as MemoryCategory,
            sourceConvId
          );
        } catch (err) {
          throw new ChatError(
            err instanceof Error ? err.message : "Could not save memory.",
            400
          );
        }

        res.status(201).json({ memory });
      } catch (error) {
        next(error);
      }
    }
  );

  app.post(
    "/api/memories/extract",
    chatLimiter,
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        if (!isPlainObject(req.body)) {
          throw new ChatError("Invalid request body.", 400);
        }
        const conversationId = validateConversationId(req.body.conversation_id);
        if (!conversationId) {
          throw new ChatError("Conversation ID is required.", 400);
        }

        const result = await engine.extractUsefulMemories(conversationId);
        res.json({
          added: result.added,
          skipped_sensitive: result.skippedSensitive,
          memories: database.getMemories(),
        });
      } catch (error) {
        next(error);
      }
    }
  );

  app.delete(
    "/api/memories/:memory_id",
    (req: Request, res: Response, next: NextFunction) => {
      try {
        const id = Number(req.params.memory_id);
        if (!Number.isInteger(id) || id <= 0) {
          throw new ChatError("Invalid memory ID.", 400);
        }
        const deleted = database.deleteMemory(id);
        if (!deleted) {
          res.status(404).json({ error: "Memory not found." });
          return;
        }
        res.json({ status: "deleted", id });
      } catch (error) {
        next(error);
      }
    }
  );

  app.delete("/api/memories", (_req: Request, res: Response) => {
    const removed = database.clearMemories();
    res.json({ status: "cleared", removed });
  });

  // Delete conversation
  app.delete(
    "/api/conversations/:conversation_id",
    (req: Request, res: Response, next: NextFunction) => {
      try {
        const conversationId = validateConversationId(req.params.conversation_id);
        if (!conversationId) {
          res.status(404).json({ error: "Conversation not found." });
          return;
        }

        const existing = database.getConversation(conversationId);
        engine.reset(conversationId);

        if (!existing) {
          res.status(404).json({ error: "Conversation not found." });
          return;
        }

        res.json({
          status: "deleted",
          conversation_id: conversationId,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  // API 404 fallback (avoid returning HTML on unknown /api routes)
  app.use("/api", (_req: Request, res: Response) => {
    res.status(404).json({ error: "API endpoint not found." });
  });

  // Error handlers
  app.use((err: unknown, req: Request, res: Response, _next: NextFunction) => {
    if (err instanceof ChatError) {
      res.status(err.status).json({ error: err.userMessage });
      return;
    }

    const safeMessage = redactSecrets(
      err instanceof Error ? err.message : String(err)
    );
    if (config.isProduction) {
      console.error(`[convoai] Unexpected error on ${req.path}: ${safeMessage}`);
    } else {
      const safeStack =
        err instanceof Error && err.stack
          ? redactSecrets(err.stack)
          : safeMessage;
      console.error(`[convoai] Unexpected error on ${req.path}: ${safeStack}`);
    }
    res.status(500).json({
      error: "Unexpected server error. Please try again.",
    });
  });

  return { app, database, engine, config };
}

const isMainEntry =
  typeof process.argv[1] === "string" &&
  path.resolve(process.argv[1]) === __filename;

if (isMainEntry) {
  const { app, database, config } = createApp();
  const server = app.listen(config.port, config.host, () => {
    console.info(
      `[convoai] Server (${config.nodeEnv}) listening on http://${config.host}:${config.port}`
    );
  });

  let shuttingDown = false;
  const shutdown = (signal: string) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.info(`[convoai] Received ${signal}, shutting down cleanly...`);
    server.close(() => {
      database.close();
      process.exit(0);
    });
    setTimeout(() => {
      database.close();
      process.exit(1);
    }, 5000).unref();
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
}
