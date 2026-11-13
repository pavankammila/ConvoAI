/**
 * ConvoAI - Chat Engine
 *
 * Responsibilities:
 * 1. Maintain separate conversations.
 * 2. Preserve multi-turn conversation history.
 * 3. Persist conversations using SQLite.
 * 4. Manage context window limits.
 * 5. Summarize older messages.
 * 6. Recover gracefully if summarization fails.
 * 7. Restore conversations after application restarts.
 * 8. Support both standard and streaming responses.
 */

import { randomUUID } from "node:crypto";
import { ChatError } from "./errors.js";
import {
  isSensitiveMemoryContent,
  type Database,
  type HistoryMessage,
  type MemoryCategory,
  type MemoryRecord,
  type MessageAttachment,
  type PersonalityId,
  type ResponseLength,
  type ResponseTone,
  type UserPreferences,
} from "./database.js";
import {
  DEFAULT_MODEL,
  redactSecrets,
  SUPPORTED_MODELS,
  type LLMClient,
} from "./llm_client.js";

export const SYSTEM_PROMPT = `You are ConvoAI, an intelligent, helpful, friendly, and professional AI assistant.

Personality & Communication:
- Answer naturally, warmly, and clearly without being condescending.
- Maintain relevant context across turns and understand follow-up questions seamlessly.
- Format complex or multi-part answers cleanly using structured Markdown (headings, bullet lists, tables, and fenced code blocks when helpful).

Capabilities:
- Explain technical and conceptual topics clearly, adapting depth to the user's needs.
- Write, review, explain, and debug code accurately, identifying root causes and practical fixes.
- Help users learn step by step with concrete examples.
- Summarize information accurately and concisely.

Honesty & Memory Accuracy:
- If you are uncertain or lack sufficient information, say so clearly and ask a brief clarifying question when helpful.
- Never fabricate facts, APIs, citations, file outputs, or technical behavior.
- Memory honesty: Only claim to know or remember information that appears in the current conversation history, the current conversation summary, or the Explicit Long-Term User Memory section below. Never pretend to remember unprovided context.
`;

export const PERSONALITY_PROMPTS: Record<
  Exclude<PersonalityId, "custom">,
  { label: string; description: string; prompt: string }
> = {
  friendly: {
    label: "Friendly",
    description: "Warm, approachable, and encouraging everyday assistant.",
    prompt:
      "Active Personality — Friendly Assistant: Be warm, approachable, supportive, and easy to talk to. Make complex topics feel welcoming.",
  },
  professional: {
    label: "Professional",
    description: "Crisp, structured, objective, and executive-ready.",
    prompt:
      "Active Personality — Professional Assistant: Communicate with polished executive clarity, structured logic, objective analysis, and well-organized Markdown deliverables.",
  },
  coding: {
    label: "Coding",
    description: "Senior software engineer for architecture, code, and debugging.",
    prompt:
      "Active Personality — Coding Assistant: Act as a senior software engineer. Prioritize idiomatic, maintainable code, root-cause debugging, security, edge cases, and clear technical trade-offs with clean code blocks.",
  },
  study: {
    label: "Study",
    description: "Patient tutor with step-by-step explanations and active recall.",
    prompt:
      "Active Personality — Study Assistant: Act as a patient, encouraging tutor. Break concepts into intuitive first-principles steps, use memorable analogies, and offer quick practice prompts or check-in questions when helpful.",
  },
  creative: {
    label: "Creative",
    description: "Imaginative writer and brainstormer with vivid craft.",
    prompt:
      "Active Personality — Creative Writing Assistant: Bring strong narrative voice, vivid imagery, original angles, and thoughtful craft while honoring genre, pacing, and user direction.",
  },
};

export const RESPONSE_LENGTH_INSTRUCTIONS: Record<ResponseLength, string> = {
  concise:
    "Response Length — Concise: Keep replies tight, high-signal, and scannable. Prefer brief paragraphs or short bullet points over lengthy prose.",
  balanced:
    "Response Length — Balanced: Provide clear, well-proportioned answers with enough explanation and examples to be thorough without unnecessary fluff.",
  detailed:
    "Response Length — Detailed: Give comprehensive, in-depth explanations covering nuances, step-by-step reasoning, edge cases, and concrete examples.",
};

export const RESPONSE_TONE_INSTRUCTIONS: Record<ResponseTone, string> = {
  warm: "Response Tone — Warm: Maintain a welcoming, empathetic, and encouraging voice.",
  neutral:
    "Response Tone — Neutral: Maintain a balanced, calm, objective, and matter-of-fact voice.",
  direct:
    "Response Tone — Direct: Get straight to the point with minimal preamble or filler.",
  playful:
    "Response Tone — Playful: Use light wit and engaging phrasing where fitting while staying accurate and helpful.",
};

export const SUMMARY_SYSTEM = `
You are a conversation summarization assistant.

Create a concise factual summary of the conversation.
Preserve:
- User goals and important facts.
- Previous decisions.
- Important technical details.
- Unresolved questions.

Maximum length: 150 words.
Output only the summary.
`;

export const MEMORY_EXTRACTION_SYSTEM = `
You analyze a conversation and extract ONLY durable, non-sensitive, genuinely useful long-term facts about the user (such as preferred name, role, technical stack, recurring preferences, learning goals, or active projects).

Rules:
- NEVER store passwords, API keys, tokens, financial details, government IDs, private addresses, or one-off ephemeral questions.
- Only include clear facts explicitly stated by the user.
- Maximum 3 short bullet-free facts (each under 120 characters).
- Respond ONLY with a valid JSON array of objects: [{"category":"preference|goal|background|project|general","content":"..."}]
- If there are no useful durable facts, respond with [].
`;

export const MAX_MESSAGE_CHARS = 4000;
export const MAX_CONVERSATIONS = 200;

interface ConversationState {
  history: HistoryMessage[];
  summary: string;
  customInstructions: string;
  personality: PersonalityId | "";
  responseLength: ResponseLength | "";
  responseTone: ResponseTone | "";
  queue: Promise<void>;
}

export interface ChatResponsePayload {
  conversation_id: string;
  reply: string;
  messages_in_context: number;
  has_summary: boolean;
}

export interface StreamHandlers {
  onMeta?: (meta: { conversation_id: string; has_summary: boolean }) => void;
  onChunk?: (chunk: string) => void;
  signal?: AbortSignal;
}

export class ChatEngine {
  private readonly llm: LLMClient;
  private readonly systemPrompt: string;
  private readonly database: Database | null;
  private readonly maxHistory: number;
  private readonly keepRecent: number;
  private readonly conversations = new Map<string, ConversationState>();

  constructor(
    llm: LLMClient,
    options: {
      systemPrompt?: string;
      maxHistory?: number;
      keepRecent?: number;
      database?: Database | null;
    } = {}
  ) {
    this.llm = llm;
    this.systemPrompt = options.systemPrompt ?? SYSTEM_PROMPT;
    this.database = options.database ?? null;

    const envMaxHistory = Number(process.env.MAX_HISTORY || "12");
    const envKeepRecent = Number(process.env.KEEP_RECENT || "6");

    this.maxHistory =
      options.maxHistory ??
      (Number.isFinite(envMaxHistory) && envMaxHistory > 0
        ? Math.floor(envMaxHistory)
        : 12);
    this.keepRecent =
      options.keepRecent ??
      (Number.isFinite(envKeepRecent) && envKeepRecent > 0
        ? Math.floor(envKeepRecent)
        : 6);

    if (this.keepRecent < 2 || this.keepRecent >= this.maxHistory) {
      throw new Error("Need 2 <= KEEP_RECENT < MAX_HISTORY");
    }
  }

  getDefaultModelName(): string {
    return typeof this.llm.getModel === "function"
      ? this.llm.getModel()
      : DEFAULT_MODEL;
  }

  async send(
    conversationId: string | null | undefined,
    rawText: string,
    attachments: MessageAttachment[] = []
  ): Promise<ChatResponsePayload> {
    return this.executeTurn(conversationId, rawText, undefined, attachments);
  }

  async sendStream(
    conversationId: string | null | undefined,
    rawText: string,
    handlers: StreamHandlers = {},
    attachments: MessageAttachment[] = []
  ): Promise<ChatResponsePayload> {
    return this.executeTurn(conversationId, rawText, handlers, attachments);
  }

  async regenerateStream(
    conversationId: string,
    handlers: StreamHandlers = {}
  ): Promise<ChatResponsePayload> {
    if (!conversationId) {
      throw new ChatError("Conversation ID is required.", 400);
    }

    const { conv } = this.getOrCreate(conversationId);
    let lastUserPrompt: string | null = null;
    let lastUserAttachments: MessageAttachment[] = [];

    // Pop last turn cleanly inside the conversation queue
    await new Promise<void>((resolve, reject) => {
      const pop = async () => {
        if (conv.history.length >= 2 && conv.history[conv.history.length - 1].role === "model") {
          conv.history.pop();
          const userMsg = conv.history.pop();
          lastUserPrompt = userMsg?.text ?? null;
          lastUserAttachments = userMsg?.attachments ? [...userMsg.attachments] : [];
        } else if (conv.history.length >= 1 && conv.history[conv.history.length - 1].role === "user") {
          const userMsg = conv.history.pop();
          lastUserPrompt = userMsg?.text ?? null;
          lastUserAttachments = userMsg?.attachments ? [...userMsg.attachments] : [];
        }

        if (this.database !== null) {
          const popped = this.database.popLastTurnWithAttachments(conversationId);
          if (popped) {
            if (!lastUserPrompt) {
              lastUserPrompt = popped.text;
            }
            if (lastUserAttachments.length === 0 && popped.attachments.length > 0) {
              lastUserAttachments = popped.attachments;
            }
          }
        }
      };

      const next = conv.queue.then(pop, pop);
      conv.queue = next.then(
        () => undefined,
        () => undefined
      );
      next.then(resolve, reject);
    });

    if (!lastUserPrompt && lastUserAttachments.length === 0) {
      throw new ChatError("No message found to regenerate.", 404);
    }

    return this.executeTurn(
      conversationId,
      lastUserPrompt ?? "",
      handlers,
      lastUserAttachments
    );
  }

  clearSummary(conversationId: string): boolean {
    const conv = this.conversations.get(conversationId);
    if (conv) {
      conv.summary = "";
    }
    if (this.database !== null) {
      return this.database.updateSummary(conversationId, "");
    }
    return Boolean(conv);
  }

  syncConversationSettings(
    conversationId: string,
    settings: {
      custom_instructions: string;
      personality: PersonalityId | "";
      response_length: ResponseLength | "";
      response_tone: ResponseTone | "";
    }
  ): void {
    const conv = this.conversations.get(conversationId);
    if (conv) {
      conv.customInstructions = settings.custom_instructions;
      conv.personality = settings.personality;
      conv.responseLength = settings.response_length;
      conv.responseTone = settings.response_tone;
    }
  }

  async extractUsefulMemories(conversationId: string): Promise<{
    added: MemoryRecord[];
    skippedSensitive: number;
  }> {
    if (!this.database) {
      throw new ChatError("Database is not configured.", 500);
    }
    const prefs = this.database.getPreferences();
    if (!prefs.consent_given || !prefs.memory_enabled) {
      throw new ChatError(
        "Enable preference consent and long-term memory in Settings before saving memories.",
        400
      );
    }

    const messages = this.database.getMessages(conversationId);
    const userMessages = messages.filter((m) => m.role === "user");
    if (userMessages.length === 0) {
      return { added: [], skippedSensitive: 0 };
    }

    const transcriptText = messages
      .slice(-12)
      .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content.slice(0, 600)}`)
      .join("\n");

    const raw = await this.llm.generate(
      MEMORY_EXTRACTION_SYSTEM,
      [{ role: "user", text: `Conversation:\n${transcriptText}` }],
      { retries: 0 }
    );

    const cleanedJson = raw
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/i, "")
      .trim();

    let parsed: unknown;
    try {
      parsed = JSON.parse(cleanedJson);
    } catch {
      return { added: [], skippedSensitive: 0 };
    }

    if (!Array.isArray(parsed)) {
      return { added: [], skippedSensitive: 0 };
    }

    const allowedCategories = new Set<MemoryCategory>([
      "preference",
      "goal",
      "background",
      "project",
      "general",
    ]);

    const added: MemoryRecord[] = [];
    let skippedSensitive = 0;

    for (const item of parsed.slice(0, 3)) {
      if (!item || typeof item !== "object") continue;
      const content =
        typeof (item as Record<string, unknown>).content === "string"
          ? ((item as Record<string, unknown>).content as string).trim()
          : "";
      const rawCat = (item as Record<string, unknown>).category;
      const category: MemoryCategory =
        typeof rawCat === "string" && allowedCategories.has(rawCat as MemoryCategory)
          ? (rawCat as MemoryCategory)
          : "general";

      if (!content) continue;
      if (isSensitiveMemoryContent(content)) {
        skippedSensitive++;
        continue;
      }

      try {
        added.push(this.database.addMemory(content, category, conversationId));
      } catch {
        skippedSensitive++;
      }
    }

    return { added, skippedSensitive };
  }

  invalidateCache(conversationId: string): void {
    this.conversations.delete(conversationId);
  }

  reset(conversationId: string | null | undefined): void {
    if (!conversationId) return;
    this.conversations.delete(conversationId);
    if (this.database !== null) {
      this.database.deleteConversation(conversationId);
    }
  }

  private async executeTurn(
    conversationId: string | null | undefined,
    rawText: string,
    streamHandlers?: StreamHandlers,
    attachments: MessageAttachment[] = []
  ): Promise<ChatResponsePayload> {
    const rawTrimmed = (rawText || "").trim();
    const cleanAttachments = Array.isArray(attachments)
      ? attachments.map((a) => ({ ...a }))
      : [];

    if (!rawTrimmed && cleanAttachments.length === 0) {
      throw new ChatError("Please type a message or attach a file first.", 400);
    }

    if (rawTrimmed.length > MAX_MESSAGE_CHARS) {
      throw new ChatError(
        `Message is too long. Maximum ${MAX_MESSAGE_CHARS} characters.`,
        400
      );
    }

    const text = rawTrimmed || "Please analyze the attached file(s).";

    const { id, conv } = this.getOrCreate(conversationId ?? null);

    let result!: ChatResponsePayload;
    const runTurn = async () => {
      const previousHistory = conv.history.map((m) => ({
        ...m,
        attachments: m.attachments ? m.attachments.map((a) => ({ ...a })) : undefined,
      }));
      const previousSummary = conv.summary;

      const userEntry: HistoryMessage = cleanAttachments.length
        ? { role: "user", text, attachments: cleanAttachments }
        : { role: "user", text };
      conv.history.push(userEntry);

      let compactedCount = 0;
      let reply: string;

      try {
        compactedCount = await this.compact(conv, streamHandlers?.signal);

        streamHandlers?.onMeta?.({
          conversation_id: id,
          has_summary: Boolean(conv.summary),
        });

        const systemInstruction = this.systemFor(conv);
        const prefs = this.database ? this.database.getPreferences() : null;
        const selectedModel =
          prefs?.consent_given &&
          prefs.model &&
          SUPPORTED_MODELS.some((m) => m.id === prefs.model)
            ? prefs.model
            : undefined;

        const historySnapshot = conv.history.map((m) => ({
          ...m,
          attachments: m.attachments
            ? m.attachments.map((a) => ({ ...a }))
            : undefined,
        }));

        let generated: string;
        if (streamHandlers?.onChunk && typeof this.llm.generateStream === "function") {
          generated = await this.llm.generateStream(
            systemInstruction,
            historySnapshot,
            streamHandlers.onChunk,
            { signal: streamHandlers.signal, model: selectedModel }
          );
        } else {
          generated = await this.llm.generate(
            systemInstruction,
            historySnapshot,
            { signal: streamHandlers?.signal, model: selectedModel }
          );
          if (generated && streamHandlers?.onChunk) {
            streamHandlers.onChunk(generated);
          }
        }

        if (typeof generated !== "string" || !generated.trim()) {
          throw new ChatError(
            "The AI returned an empty response. Please try again.",
            502
          );
        }

        reply = generated.trim();
      } catch (error) {
        conv.history = previousHistory;
        conv.summary = previousSummary;
        throw error;
      }

      conv.history.push({ role: "model", text: reply });

      if (this.database !== null) {
        try {
          this.database.saveTurnState(
            id,
            text,
            reply,
            compactedCount,
            conv.summary,
            cleanAttachments
          );
        } catch (error) {
          conv.history = previousHistory;
          conv.summary = previousSummary;
          const msg = redactSecrets(
            error instanceof Error ? error.message : String(error)
          );
          console.error(`[convoai] Failed to persist conversation state: ${msg}`);
          throw new ChatError(
            "Your response was generated, but saving the conversation failed.",
            500
          );
        }
      }

      result = {
        conversation_id: id,
        reply,
        messages_in_context: conv.history.length,
        has_summary: Boolean(conv.summary),
      };
    };

    const next = conv.queue.then(runTurn, runTurn);
    conv.queue = next.then(
      () => undefined,
      () => undefined
    );

    await next;
    return result;
  }

  private getOrCreate(conversationId: string | null): {
    id: string;
    conv: ConversationState;
  } {
    if (conversationId && this.conversations.has(conversationId)) {
      const existing = this.conversations.get(conversationId)!;
      this.conversations.delete(conversationId);
      this.conversations.set(conversationId, existing);
      return { id: conversationId, conv: existing };
    }

    const newId = conversationId || randomUUID().replace(/-/g, "");
    const conv: ConversationState = {
      history: [],
      summary: "",
      customInstructions: "",
      personality: "",
      responseLength: "",
      responseTone: "",
      queue: Promise.resolve(),
    };

    if (this.database !== null) {
      const saved = this.database.getConversation(newId);
      if (saved) {
        conv.summary = saved.summary;
        conv.customInstructions = saved.custom_instructions;
        conv.personality = saved.personality;
        conv.responseLength = saved.response_length;
        conv.responseTone = saved.response_tone;
        const savedMessages = this.database.getContextMessages(newId);
        conv.history = savedMessages.map((msg) =>
          msg.attachments && msg.attachments.length > 0
            ? { role: msg.role, text: msg.content, attachments: msg.attachments }
            : { role: msg.role, text: msg.content }
        );
      } else {
        this.database.createConversation(newId);
      }
    }

    this.conversations.set(newId, conv);

    while (this.conversations.size > MAX_CONVERSATIONS) {
      const oldestKey = this.conversations.keys().next().value;
      if (oldestKey === undefined) break;
      this.conversations.delete(oldestKey);
    }

    return { id: newId, conv };
  }

  private systemFor(conv: ConversationState): string {
    const sections: string[] = [this.systemPrompt.trim()];

    const prefs: UserPreferences | null = this.database
      ? this.database.getPreferences()
      : null;

    const activePersonality: PersonalityId =
      conv.personality || prefs?.personality || "friendly";
    const activeLength: ResponseLength =
      conv.responseLength || prefs?.response_length || "balanced";
    const activeTone: ResponseTone =
      conv.responseTone || prefs?.response_tone || "neutral";

    if (activePersonality === "custom") {
      const customPrompt = prefs?.custom_system_prompt?.trim();
      if (customPrompt) {
        sections.push(`Active Custom System Personality:\n${customPrompt}`);
      }
    } else {
      const preset = PERSONALITY_PROMPTS[activePersonality];
      if (preset) {
        sections.push(preset.prompt);
      }
    }

    sections.push(RESPONSE_LENGTH_INSTRUCTIONS[activeLength]);
    sections.push(RESPONSE_TONE_INSTRUCTIONS[activeTone]);

    if (prefs?.consent_given && prefs.preferred_name) {
      sections.push(`User's preferred name: ${prefs.preferred_name}`);
    }

    if (prefs?.consent_given && prefs.memory_enabled && this.database) {
      const memories = this.database.getMemories();
      if (memories.length > 0) {
        const memLines = memories
          .slice(0, 25)
          .map((m) => `- [${m.category}] ${m.content}`)
          .join("\n");
        sections.push(
          `Explicit Long-Term User Memory (facts explicitly saved by the user; do not claim to know anything beyond these and the current conversation):\n${memLines}`
        );
      }
    }

    if (conv.customInstructions) {
      sections.push(
        `Conversation-Specific Instructions (apply to this conversation only):\n${conv.customInstructions}`
      );
    }

    if (conv.summary) {
      sections.push(
        `Earlier conversation summary (this conversation's short-term history):\n${conv.summary}\nUse this summary as background context.`
      );
    }

    return sections.join("\n\n");
  }

  private async compact(
    conv: ConversationState,
    signal?: AbortSignal
  ): Promise<number> {
    if (conv.history.length <= this.maxHistory) {
      return 0;
    }

    const cut = conv.history.length - this.keepRecent;
    const old = conv.history.slice(0, cut);
    const recent = conv.history.slice(cut);

    while (recent.length > 0 && recent[0].role !== "user") {
      const shifted = recent.shift();
      if (shifted) old.push(shifted);
    }

    try {
      const newSummary = await this.summarize(conv.summary, old, signal);
      conv.summary = newSummary;
    } catch (error) {
      if (signal?.aborted) {
        throw error;
      }
      console.warn(
        `[convoai] Summarization failed. Falling back to truncation: ${redactSecrets(
          error instanceof Error ? error.message : String(error)
        )}`
      );
    }

    conv.history = recent;
    return old.length;
  }

  private async summarize(
    previousSummary: string,
    old: HistoryMessage[],
    signal?: AbortSignal
  ): Promise<string> {
    const lines = old.map((msg) => {
      const role = msg.role === "user" ? "User" : "Assistant";
      const attNote =
        msg.attachments && msg.attachments.length > 0
          ? ` [Attached: ${msg.attachments.map((a) => a.name).join(", ")}]`
          : "";
      return `${role}:${attNote} ${msg.text.slice(0, 1000)}`.trim();
    });

    const prompt =
      `Previous summary:\n${previousSummary || "(none)"}\n\n` +
      `New conversation messages:\n${lines.join("\n")}\n\n` +
      `Create an updated summary.`;

    // Single-attempt summary call so fallback truncation happens quickly on errors
    const result = await this.llm.generate(
      SUMMARY_SYSTEM,
      [{ role: "user", text: prompt }],
      { retries: 0, signal }
    );

    if (typeof result !== "string" || !result.trim()) {
      throw new Error("Empty summary returned by AI.");
    }

    return result.trim().slice(0, 2000);
  }
}
