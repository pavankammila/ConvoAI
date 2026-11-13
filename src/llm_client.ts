/**
 * ConvoAI - Gemini API Client
 *
 * Server-side Google Gen AI SDK integration with timeouts,
 * streaming support, automatic retries, exponential backoff,
 * and friendly API error messages.
 */

import { GoogleGenAI, type GenerateContentResponse } from "@google/genai";
import { ChatError } from "./errors.js";
import type { HistoryMessage } from "./database.js";

export interface GenerateOptions {
  retries?: number;
  signal?: AbortSignal;
  model?: string;
}

export interface LLMClient {
  generate(
    system: string,
    messages: HistoryMessage[],
    options?: GenerateOptions
  ): Promise<string>;
  generateStream?(
    system: string,
    messages: HistoryMessage[],
    onChunk: (chunk: string) => void,
    options?: GenerateOptions
  ): Promise<string>;
  getModel?(): string;
}

export type GeminiPart =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } };

export interface GoogleGenAIClientLike {
  models: {
    generateContent(params: {
      model: string;
      contents: Array<{ role: string; parts: GeminiPart[] }>;
      config?: { systemInstruction?: string; abortSignal?: AbortSignal };
    }): Promise<GenerateContentResponse | Record<string, unknown>>;
    generateContentStream(params: {
      model: string;
      contents: Array<{ role: string; parts: GeminiPart[] }>;
      config?: { systemInstruction?: string; abortSignal?: AbortSignal };
    }): Promise<AsyncIterable<GenerateContentResponse | Record<string, unknown>>>;
  };
}

export interface SupportedModelInfo {
  id: string;
  label: string;
  description: string;
}

export const DEFAULT_MODEL = "gemini-4-hydrogen";

export const SUPPORTED_MODELS: SupportedModelInfo[] = [
  {
    id: "gemini-4-hydrogen",
    label: "Gemini 4 Hydrogen",
    description: "Balanced reasoning, coding, and multimodal understanding.",
  },
  {
    id: "gemini-3.1-flash-lite",
    label: "Gemini 3.1 Flash Lite",
    description: "Fast, lightweight responses with multimodal support.",
  },
];

export function buildContentParts(msg: HistoryMessage): GeminiPart[] {
  const parts: GeminiPart[] = [];
  const textBlocks: string[] = [];

  if (msg.text && msg.text.trim()) {
    textBlocks.push(msg.text);
  }

  for (const att of msg.attachments ?? []) {
    if (att.kind === "text" && typeof att.textContent === "string") {
      textBlocks.push(
        `--- Attached Document: ${att.name} (${att.mimeType}) ---\n${att.textContent}\n--- End of ${att.name} ---`
      );
    } else if ((att.kind === "image" || att.kind === "pdf") && att.data) {
      parts.push({
        inlineData: {
          mimeType: att.mimeType,
          data: att.data,
        },
      });
    }
  }

  if (textBlocks.length > 0) {
    parts.unshift({ text: textBlocks.join("\n\n") });
  } else if (parts.length === 0) {
    parts.push({ text: msg.text || "" });
  }

  return parts;
}

const BLOCKED_REASONS = new Set([
  "SAFETY",
  "PROHIBITED_CONTENT",
  "BLOCKLIST",
  "SPII",
  "RECITATION",
  "IMAGE_SAFETY",
  "OTHER",
]);

export function redactSecrets(text: string, apiKey?: string): string {
  let sanitized = text;
  if (apiKey) {
    sanitized = sanitized.split(apiKey).join("[REDACTED]");
  }
  const envKey = (process.env.GEMINI_API_KEY ?? "").trim();
  if (envKey) {
    sanitized = sanitized.split(envKey).join("[REDACTED]");
  }
  sanitized = sanitized.replace(/AIza[0-9A-Za-z_-]{16,}/g, "[REDACTED_KEY]");
  sanitized = sanitized.replace(/\bsk-(?:proj-)?[0-9A-Za-z_-]{16,}\b/g, "[REDACTED_KEY]");
  sanitized = sanitized.replace(/\bBearer\s+[A-Za-z0-9._~+/=-]{12,}/gi, "Bearer [REDACTED]");
  sanitized = sanitized.replace(/([?&]key=)[^&\s]+/gi, "$1[REDACTED]");
  return sanitized;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new ChatError("Generation cancelled.", 499));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new ChatError("Generation cancelled.", 499));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export function extractStatusCode(error: unknown): number | null {
  if (!error || typeof error !== "object") return null;
  const err = error as Record<string, unknown>;
  if (typeof err.status === "number") return err.status;
  if (typeof err.code === "number") return err.code;
  const msg = typeof err.message === "string" ? err.message : "";
  const match = msg.match(/\b(400|401|403|404|429|500|502|503|504)\b/);
  return match ? Number(match[1]) : null;
}

export function mapApiError(code: number | null, message: string): ChatError {
  const lowered = message.toLowerCase();

  if (
    code === 401 ||
    code === 403 ||
    lowered.includes("api key") ||
    lowered.includes("api_key_invalid") ||
    lowered.includes("permission_denied")
  ) {
    return new ChatError(
      "The API key was rejected. Check GEMINI_API_KEY in Settings > Secrets.",
      502
    );
  }

  if (
    code === 404 ||
    lowered.includes("not_found") ||
    (lowered.includes("model") &&
      (lowered.includes("not found") ||
        lowered.includes("not supported") ||
        lowered.includes("unsupported")))
  ) {
    return new ChatError(
      "The Gemini model was not found or is unsupported. Check your GEMINI_MODEL setting.",
      502
    );
  }

  if (
    code === 429 ||
    lowered.includes("resource_exhausted") ||
    lowered.includes("rate limit") ||
    lowered.includes("quota")
  ) {
    return new ChatError(
      "Gemini rate limit reached. Please wait a few seconds and try again.",
      429
    );
  }

  if (
    (code !== null && [500, 502, 503, 504].includes(code)) ||
    lowered.includes("unavailable") ||
    lowered.includes("overloaded")
  ) {
    return new ChatError(
      "The Gemini service is temporarily busy. Please try again shortly.",
      503
    );
  }

  if (code === 400) {
    return new ChatError(
      "The request could not be processed by Gemini. Check your input and configuration.",
      400
    );
  }

  return new ChatError(
    "Something went wrong while communicating with Gemini.",
    502
  );
}

function getBlockReason(response: Record<string, unknown>): string | null {
  const feedback = response.promptFeedback;
  if (feedback && typeof feedback === "object") {
    const reason = (feedback as Record<string, unknown>).blockReason;
    if (typeof reason === "string" && reason && reason !== "BLOCK_REASON_UNSPECIFIED") {
      return reason;
    }
  }

  const candidates = response.candidates;
  if (Array.isArray(candidates) && candidates.length > 0) {
    const first = candidates[0];
    if (first && typeof first === "object") {
      const finishReason = (first as Record<string, unknown>).finishReason;
      if (typeof finishReason === "string" && BLOCKED_REASONS.has(finishReason)) {
        return finishReason;
      }
    }
  }

  return null;
}

function extractTextFromResponse(response: unknown): {
  text: string;
  blockReason: string | null;
} {
  if (!response || typeof response !== "object" || Array.isArray(response)) {
    throw new ChatError(
      "Received a malformed response from Gemini. Please try again.",
      502
    );
  }

  const record = response as Record<string, unknown>;
  const blockReason = getBlockReason(record);

  let rawText: unknown;
  try {
    rawText = record.text;
  } catch {
    throw new ChatError(
      "Received a malformed response from Gemini. Please try again.",
      502
    );
  }

  if (typeof rawText === "string") {
    return { text: rawText, blockReason };
  }

  if (rawText !== undefined && rawText !== null) {
    throw new ChatError(
      "Received a malformed response from Gemini. Please try again.",
      502
    );
  }

  // Fallback inspection of candidates[0].content.parts when `.text` is undefined
  if (record.candidates !== undefined) {
    if (!Array.isArray(record.candidates)) {
      throw new ChatError(
        "Received a malformed response from Gemini. Please try again.",
        502
      );
    }
    const first = record.candidates[0];
    if (first && typeof first === "object") {
      const content = (first as Record<string, unknown>).content;
      if (content && typeof content === "object") {
        const parts = (content as Record<string, unknown>).parts;
        if (Array.isArray(parts)) {
          const joined = parts
            .map((p) =>
              p && typeof p === "object" && typeof (p as Record<string, unknown>).text === "string"
                ? ((p as Record<string, unknown>).text as string)
                : ""
            )
            .join("");
          return { text: joined, blockReason };
        }
      }
    }
  }

  return { text: "", blockReason };
}

export class GeminiClient implements LLMClient {
  private readonly client: GoogleGenAIClientLike;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly retries: number;
  private readonly backoffMs: number;
  private readonly timeoutMs: number;

  constructor(options: {
    apiKey: string;
    model?: string;
    timeoutSeconds?: number;
    retries?: number;
    backoffSeconds?: number;
    client?: GoogleGenAIClientLike;
  }) {
    const cleanKey = (options.apiKey ?? "").trim();
    if (!cleanKey && !options.client) {
      throw new ChatError(
        "Gemini API key is missing. Please add GEMINI_API_KEY in Settings > Secrets.",
        503
      );
    }

    this.apiKey = cleanKey;
    this.model = (options.model ?? "").trim() || DEFAULT_MODEL;
    this.retries = Math.max(0, Math.min(5, options.retries ?? 2));
    this.backoffMs = Math.max(0, (options.backoffSeconds ?? 1.0) * 1000);
    this.timeoutMs = Math.max(
      100,
      Math.round((options.timeoutSeconds ?? 30) * 1000)
    );

    this.client =
      options.client ??
      (new GoogleGenAI({
        apiKey: cleanKey,
        httpOptions: {
          timeout: this.timeoutMs,
          headers: {
            "User-Agent": "aistudio-build",
          },
        },
      }) as unknown as GoogleGenAIClientLike);
  }

  getModel(): string {
    return this.model;
  }

  getTimeoutMs(): number {
    return this.timeoutMs;
  }

  private buildAttemptSignal(userSignal?: AbortSignal): {
    signal: AbortSignal;
    timedOut: () => boolean;
    cleanup: () => void;
  } {
    const timeoutController = new AbortController();
    let didTimeout = false;
    const timer = setTimeout(() => {
      didTimeout = true;
      timeoutController.abort(new Error("Request timeout"));
    }, this.timeoutMs);

    const signal = userSignal
      ? AbortSignal.any([userSignal, timeoutController.signal])
      : timeoutController.signal;

    return {
      signal,
      timedOut: () => didTimeout,
      cleanup: () => clearTimeout(timer),
    };
  }

  private classifyError(
    error: unknown,
    attempt: number,
    timedOut: boolean
  ): {
    chatError: ChatError;
    retryable: boolean;
  } {
    const rawMessage = error instanceof Error ? error.message : String(error);
    const safeMessage = redactSecrets(rawMessage, this.apiKey);
    const lowered = safeMessage.toLowerCase();
    const code = extractStatusCode(error);

    if (
      timedOut ||
      lowered.includes("timeout") ||
      lowered.includes("timed out") ||
      lowered.includes("etimedout") ||
      lowered.includes("und_err_headers_timeout") ||
      lowered.includes("und_err_connect_timeout")
    ) {
      console.warn(`[convoai] Gemini request timed out (attempt ${attempt + 1})`);
      return {
        chatError: new ChatError(
          "The AI took too long to respond. Please try again.",
          504
        ),
        retryable: true,
      };
    }

    if (
      lowered.includes("fetch failed") ||
      lowered.includes("econnrefused") ||
      lowered.includes("econnreset") ||
      lowered.includes("enotfound") ||
      lowered.includes("eai_again") ||
      lowered.includes("network")
    ) {
      console.warn(
        `[convoai] Network error (attempt ${attempt + 1}): ${safeMessage}`
      );
      return {
        chatError: new ChatError(
          "Couldn't reach the AI service. Check your internet connection and try again.",
          503
        ),
        retryable: true,
      };
    }

    console.warn(
      `[convoai] Gemini API error (attempt ${attempt + 1}): ${safeMessage}`
    );
    const chatError = mapApiError(code, safeMessage);
    // Only retry temporary server failures (5xx). Never retry rate limits (429), auth (401/403), or bad model (404).
    const retryable = code !== null && [500, 502, 503, 504].includes(code);
    return { chatError, retryable };
  }

  async generate(
    system: string,
    messages: HistoryMessage[],
    options: GenerateOptions = {}
  ): Promise<string> {
    const activeModel = (options.model ?? "").trim() || this.model;
    const contents = messages.map((msg) => ({
      role: msg.role,
      parts: buildContentParts(msg),
    }));

    const maxRetries = Math.max(0, options.retries ?? this.retries);
    let lastError: ChatError | null = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      if (options.signal?.aborted) {
        throw new ChatError("Generation cancelled.", 499);
      }

      const attemptAbort = this.buildAttemptSignal(options.signal);

      try {
        console.info(`[convoai] Sending request to Gemini model: ${activeModel}`);

        const response = await this.client.models.generateContent({
          model: activeModel,
          contents,
          config: {
            systemInstruction: system,
            abortSignal: attemptAbort.signal,
          },
        });

        const { text, blockReason } = extractTextFromResponse(response);
        const reply = text.trim();

        if (!reply) {
          if (blockReason) {
            throw new ChatError(
              "The AI could not respond to that request due to content safety guidelines. Please try rephrasing.",
              422
            );
          }
          throw new ChatError(
            "The AI returned an empty response. Please try rephrasing your message.",
            502
          );
        }

        console.info("[convoai] Gemini response received successfully.");
        return reply;
      } catch (error: unknown) {
        if (options.signal?.aborted) {
          throw new ChatError("Generation cancelled.", 499);
        }
        if (error instanceof ChatError) {
          throw error;
        }

        const classified = this.classifyError(
          error,
          attempt,
          attemptAbort.timedOut()
        );
        lastError = classified.chatError;
        if (!classified.retryable) {
          throw lastError;
        }
      } finally {
        attemptAbort.cleanup();
      }

      if (attempt < maxRetries) {
        const delayMs = this.backoffMs * 2 ** attempt;
        if (delayMs > 0) {
          console.info(`[convoai] Retrying Gemini request in ${delayMs / 1000}s...`);
          await sleep(delayMs, options.signal);
        }
      }
    }

    if (lastError) {
      throw lastError;
    }

    throw new ChatError(
      "Unable to generate an AI response. Please try again.",
      502
    );
  }

  async generateStream(
    system: string,
    messages: HistoryMessage[],
    onChunk: (chunk: string) => void,
    options: GenerateOptions = {}
  ): Promise<string> {
    const activeModel = (options.model ?? "").trim() || this.model;
    const contents = messages.map((msg) => ({
      role: msg.role,
      parts: buildContentParts(msg),
    }));

    const maxRetries = Math.max(0, options.retries ?? this.retries);
    let lastError: ChatError | null = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      if (options.signal?.aborted) {
        throw new ChatError("Generation cancelled.", 499);
      }

      let fullText = "";
      let emittedAny = false;
      let lastBlockReason: string | null = null;
      const attemptAbort = this.buildAttemptSignal(options.signal);

      try {
        console.info(
          `[convoai] Streaming request to Gemini model: ${activeModel}`
        );

        const stream = await this.client.models.generateContentStream({
          model: activeModel,
          contents,
          config: {
            systemInstruction: system,
            abortSignal: attemptAbort.signal,
          },
        });

        if (!stream || typeof (stream as AsyncIterable<unknown>)[Symbol.asyncIterator] !== "function") {
          throw new ChatError(
            "Received a malformed stream from Gemini. Please try again.",
            502
          );
        }

        for await (const chunk of stream) {
          if (options.signal?.aborted) {
            throw new ChatError("Generation cancelled.", 499);
          }

          const { text, blockReason } = extractTextFromResponse(chunk);
          if (blockReason) {
            lastBlockReason = blockReason;
          }
          if (text) {
            fullText += text;
            emittedAny = true;
            onChunk(text);
          }
        }

        const reply = fullText.trim();
        if (!reply) {
          if (lastBlockReason) {
            throw new ChatError(
              "The AI could not respond to that request due to content safety guidelines. Please try rephrasing.",
              422
            );
          }
          throw new ChatError(
            "The AI returned an empty response. Please try rephrasing your message.",
            502
          );
        }

        console.info("[convoai] Gemini stream completed successfully.");
        return reply;
      } catch (error: unknown) {
        if (options.signal?.aborted) {
          throw new ChatError("Generation cancelled.", 499);
        }
        if (error instanceof ChatError) {
          throw error;
        }

        const classified = this.classifyError(
          error,
          attempt,
          attemptAbort.timedOut()
        );
        lastError = classified.chatError;
        if (emittedAny || !classified.retryable) {
          throw lastError;
        }
      } finally {
        attemptAbort.cleanup();
      }

      if (attempt < maxRetries) {
        const delayMs = this.backoffMs * 2 ** attempt;
        if (delayMs > 0) {
          console.info(`[convoai] Retrying Gemini stream in ${delayMs / 1000}s...`);
          await sleep(delayMs, options.signal);
        }
      }
    }

    if (lastError) {
      throw lastError;
    }

    throw new ChatError(
      "Unable to generate an AI response. Please try again.",
      502
    );
  }
}

export class UnconfiguredClient implements LLMClient {
  async generate(): Promise<string> {
    throw new ChatError(
      "Gemini API key is missing. Please add GEMINI_API_KEY in Settings > Secrets.",
      503
    );
  }

  async generateStream(): Promise<string> {
    throw new ChatError(
      "Gemini API key is missing. Please add GEMINI_API_KEY in Settings > Secrets.",
      503
    );
  }
}

export function buildClient(): LLMClient {
  const apiKey = (process.env.GEMINI_API_KEY ?? "").trim();

  if (
    !apiKey ||
    apiKey === "your-key-here" ||
    apiKey === "your_gemini_api_key"
  ) {
    console.warn("[convoai] Gemini API key is not configured.");
    return new UnconfiguredClient();
  }

  const rawModel = (process.env.GEMINI_MODEL ?? "").trim();
  const isPlaceholderModel =
    !rawModel ||
    rawModel === "your_model_name" ||
    rawModel === "your-model-name" ||
    rawModel === "your_gemini_model";
  const model = isPlaceholderModel ? DEFAULT_MODEL : rawModel;
  const rawTimeout = Number(process.env.REQUEST_TIMEOUT || "30");
  const timeoutSeconds =
    Number.isFinite(rawTimeout) && rawTimeout > 0 ? rawTimeout : 30;

  return new GeminiClient({
    apiKey,
    model,
    timeoutSeconds,
  });
}
