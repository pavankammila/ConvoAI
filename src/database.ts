/**
 * ConvoAI - SQLite Database Manager (Node.js built-in node:sqlite)
 *
 * Responsibilities:
 * 1. Store conversations permanently in SQLite.
 * 2. Store full user and assistant message transcripts + active LLM context flags.
 * 3. Store conversation summaries.
 * 4. Retrieve full transcripts and active context history.
 * 5. Search, rename, list, and delete conversations.
 * 6. Save compacted conversation state atomically inside transactions.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const DEFAULT_DATABASE_PATH = path.resolve(__dirname, "..", "convoai.db");

export type MessageRole = "user" | "model";

export type AttachmentKind = "image" | "text" | "pdf";

export interface MessageAttachment {
  kind: AttachmentKind;
  name: string;
  mimeType: string;
  data?: string;
  textContent?: string;
}

export interface StoredMessage {
  id?: number;
  role: MessageRole;
  content: string;
  attachments: MessageAttachment[];
  in_context: boolean;
  created_at: string;
}

export type PersonalityId =
  | "friendly"
  | "professional"
  | "coding"
  | "study"
  | "creative"
  | "custom";

export type ResponseLength = "concise" | "balanced" | "detailed";
export type ResponseTone = "warm" | "neutral" | "direct" | "playful";
export type MemoryCategory =
  | "preference"
  | "goal"
  | "background"
  | "project"
  | "general";

export interface ConversationRecord {
  id: string;
  title: string;
  summary: string;
  custom_instructions: string;
  personality: PersonalityId | "";
  response_length: ResponseLength | "";
  response_tone: ResponseTone | "";
  created_at: string;
  updated_at: string;
}

export interface ConversationSummaryItem {
  id: string;
  title: string;
  has_summary: boolean;
  has_custom_instructions: boolean;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface UserPreferences {
  consent_given: boolean;
  preferred_name: string;
  personality: PersonalityId;
  custom_system_prompt: string;
  response_length: ResponseLength;
  response_tone: ResponseTone;
  memory_enabled: boolean;
  model: string;
  updated_at: string;
}

export interface MemoryRecord {
  id: number;
  content: string;
  category: MemoryCategory;
  source_conversation_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface HistoryMessage {
  role: MessageRole;
  text: string;
  attachments?: MessageAttachment[];
}

interface RawConversationRow {
  id: string;
  title: string;
  summary: string;
  custom_instructions: string;
  personality: string;
  response_length: string;
  response_tone: string;
  created_at: string;
  updated_at: string;
}

interface RawMessageRow {
  id: number;
  role: string;
  content: string;
  attachments: string;
  in_context: number;
  created_at: string;
}

interface RawPreferencesRow {
  id: string;
  consent_given: number;
  preferred_name: string;
  personality: string;
  custom_system_prompt: string;
  response_length: string;
  response_tone: string;
  memory_enabled: number;
  model: string;
  updated_at: string;
}

interface RawMemoryRow {
  id: number;
  content: string;
  category: string;
  source_conversation_id: string | null;
  created_at: string;
  updated_at: string;
}

export function isSensitiveMemoryContent(text: string): boolean {
  const value = text.trim();
  if (!value) return false;
  // Detect API keys, bearer tokens, passwords, SSNs, credit cards, private keys
  if (/AIza[0-9A-Za-z_-]{16,}/.test(value)) return true;
  if (/\bsk-(?:proj-)?[0-9A-Za-z_-]{16,}\b/.test(value)) return true;
  if (/\bgh[pousr]_[0-9A-Za-z]{20,}\b/.test(value)) return true;
  if (/-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/i.test(value)) return true;
  if (/\b(?:password|passcode|pin\s*code|secret\s*key|api\s*key|access\s*token)\s*(?:is|=|:)\s*\S+/i.test(value)) {
    return true;
  }
  if (/\b\d{3}-\d{2}-\d{4}\b/.test(value)) return true;
  if (/\b(?:\d[ -]*?){13,19}\b/.test(value)) return true;
  return false;
}

export class Database {
  public readonly databasePath: string;
  private readonly db: DatabaseSync;

  constructor(databasePath: string = DEFAULT_DATABASE_PATH) {
    this.databasePath = databasePath;
    if (this.databasePath !== ":memory:") {
      fs.mkdirSync(path.dirname(this.databasePath), { recursive: true });
    }
    this.db = new DatabaseSync(this.databasePath);
    this.initialize();
  }

  static now(): string {
    return new Date().toISOString();
  }

  private initialize(): void {
    this.db.exec("PRAGMA foreign_keys = ON;");
    this.db.exec("PRAGMA busy_timeout = 5000;");
    this.db.exec("PRAGMA journal_mode = WAL;");
    this.db.exec("PRAGMA synchronous = NORMAL;");

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT 'New Chat',
        summary TEXT NOT NULL DEFAULT '',
        custom_instructions TEXT NOT NULL DEFAULT '',
        personality TEXT NOT NULL DEFAULT '',
        response_length TEXT NOT NULL DEFAULT '',
        response_tone TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'model')),
        content TEXT NOT NULL,
        attachments TEXT NOT NULL DEFAULT '[]',
        in_context INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        FOREIGN KEY (conversation_id)
          REFERENCES conversations(id)
          ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS user_preferences (
        id TEXT PRIMARY KEY,
        consent_given INTEGER NOT NULL DEFAULT 0,
        preferred_name TEXT NOT NULL DEFAULT '',
        personality TEXT NOT NULL DEFAULT 'friendly',
        custom_system_prompt TEXT NOT NULL DEFAULT '',
        response_length TEXT NOT NULL DEFAULT 'balanced',
        response_tone TEXT NOT NULL DEFAULT 'neutral',
        memory_enabled INTEGER NOT NULL DEFAULT 0,
        model TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'general',
        source_conversation_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
    `);

    const convCols = this.db
      .prepare("PRAGMA table_info(conversations)")
      .all() as Array<{ name: string }>;
    if (!convCols.some((c) => c.name === "summary")) {
      this.db.exec(
        "ALTER TABLE conversations ADD COLUMN summary TEXT NOT NULL DEFAULT '';"
      );
    }
    if (!convCols.some((c) => c.name === "custom_instructions")) {
      this.db.exec(
        "ALTER TABLE conversations ADD COLUMN custom_instructions TEXT NOT NULL DEFAULT '';"
      );
    }
    if (!convCols.some((c) => c.name === "personality")) {
      this.db.exec(
        "ALTER TABLE conversations ADD COLUMN personality TEXT NOT NULL DEFAULT '';"
      );
    }
    if (!convCols.some((c) => c.name === "response_length")) {
      this.db.exec(
        "ALTER TABLE conversations ADD COLUMN response_length TEXT NOT NULL DEFAULT '';"
      );
    }
    if (!convCols.some((c) => c.name === "response_tone")) {
      this.db.exec(
        "ALTER TABLE conversations ADD COLUMN response_tone TEXT NOT NULL DEFAULT '';"
      );
    }

    const msgCols = this.db
      .prepare("PRAGMA table_info(messages)")
      .all() as Array<{ name: string }>;
    if (!msgCols.some((c) => c.name === "in_context")) {
      this.db.exec(
        "ALTER TABLE messages ADD COLUMN in_context INTEGER NOT NULL DEFAULT 1;"
      );
    }
    if (!msgCols.some((c) => c.name === "attachments")) {
      this.db.exec(
        "ALTER TABLE messages ADD COLUMN attachments TEXT NOT NULL DEFAULT '[]';"
      );
    }

    const prefCols = this.db
      .prepare("PRAGMA table_info(user_preferences)")
      .all() as Array<{ name: string }>;
    if (!prefCols.some((c) => c.name === "model")) {
      this.db.exec(
        "ALTER TABLE user_preferences ADD COLUMN model TEXT NOT NULL DEFAULT '';"
      );
    }

    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_messages_conversation
      ON messages(conversation_id, id);
    `);
  }

  private static parseAttachments(raw: string | undefined): MessageAttachment[] {
    if (!raw) return [];
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? (parsed as MessageAttachment[]) : [];
    } catch {
      return [];
    }
  }

  private runInTransaction<T>(fn: () => T): T {
    this.db.exec("BEGIN IMMEDIATE;");
    try {
      const result = fn();
      this.db.exec("COMMIT;");
      return result;
    } catch (error) {
      try {
        this.db.exec("ROLLBACK;");
      } catch {
        // Ignore rollback error if transaction wasn't active
      }
      throw error;
    }
  }

  createConversation(conversationId: string, title = "New Chat"): void {
    const now = Database.now();
    this.db
      .prepare(
        `INSERT OR IGNORE INTO conversations (id, title, summary, created_at, updated_at)
         VALUES (?, ?, '', ?, ?)`
      )
      .run(conversationId, title, now, now);
  }

  addMessage(
    conversationId: string,
    role: MessageRole,
    content: string,
    attachments: MessageAttachment[] = []
  ): void {
    if (role !== "user" && role !== "model") {
      throw new Error("Invalid message role.");
    }

    const now = Database.now();
    const serializedAttachments = JSON.stringify(attachments ?? []);
    this.runInTransaction(() => {
      const conv = this.db
        .prepare("SELECT id, title FROM conversations WHERE id = ?")
        .get(conversationId) as { id: string; title: string } | undefined;

      if (!conv) {
        throw new Error("Conversation does not exist.");
      }

      const info = this.db
        .prepare(
          `INSERT INTO messages (conversation_id, role, content, attachments, in_context, created_at)
           VALUES (?, ?, ?, ?, 1, ?)`
        )
        .run(conversationId, role, content, serializedAttachments, now);

      if (info.changes !== 1) {
        throw new Error("Message could not be saved.");
      }

      if (role === "user" && conv.title === "New Chat") {
        const nextTitle =
          content.trim().slice(0, 50) ||
          (attachments[0]?.name ? `Attachment: ${attachments[0].name}`.slice(0, 50) : "New Chat");
        this.db
          .prepare("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?")
          .run(nextTitle, now, conversationId);
      } else {
        this.db
          .prepare("UPDATE conversations SET updated_at = ? WHERE id = ?")
          .run(now, conversationId);
      }
    });
  }

  /**
   * Atomically persist active context messages, summary, and mark compacted
   * messages as archived (`in_context = 0`) so full display history stays intact.
   */
  saveTurnState(
    conversationId: string,
    userText: string,
    modelText: string,
    compactedCount: number,
    summary: string,
    userAttachments: MessageAttachment[] = []
  ): void {
    const now = Database.now();
    const serializedUserAtt = JSON.stringify(userAttachments ?? []);

    this.runInTransaction(() => {
      const conv = this.db
        .prepare("SELECT id, title FROM conversations WHERE id = ?")
        .get(conversationId) as { id: string; title: string } | undefined;

      if (!conv) {
        throw new Error("Conversation does not exist.");
      }

      if (compactedCount > 0) {
        const activeIds = this.db
          .prepare(
            `SELECT id FROM messages
             WHERE conversation_id = ? AND in_context = 1
             ORDER BY id ASC
             LIMIT ?`
          )
          .all(conversationId, compactedCount) as Array<{ id: number }>;

        if (activeIds.length > 0) {
          const markStmt = this.db.prepare(
            "UPDATE messages SET in_context = 0 WHERE id = ?"
          );
          for (const row of activeIds) {
            markStmt.run(row.id);
          }
        }
      }

      const insertStmt = this.db.prepare(
        `INSERT INTO messages (conversation_id, role, content, attachments, in_context, created_at)
         VALUES (?, ?, ?, ?, 1, ?)`
      );
      insertStmt.run(conversationId, "user", userText, serializedUserAtt, now);
      insertStmt.run(conversationId, "model", modelText, "[]", now);

      let title = conv.title;
      if (title === "New Chat") {
        title =
          userText.trim().slice(0, 50) ||
          (userAttachments[0]?.name
            ? `Attachment: ${userAttachments[0].name}`.slice(0, 50)
            : "New Chat");
      }

      this.db
        .prepare(
          `UPDATE conversations
           SET title = ?, summary = ?, updated_at = ?
           WHERE id = ?`
        )
        .run(title, summary || "", now, conversationId);
    });
  }

  /**
   * Atomically replace stored active context messages and summary.
   * Preserves archived (`in_context = 0`) transcript rows.
   */
  saveConversationState(
    conversationId: string,
    messages: HistoryMessage[],
    summary: string
  ): void {
    const now = Database.now();

    this.runInTransaction(() => {
      const conv = this.db
        .prepare("SELECT id, title FROM conversations WHERE id = ?")
        .get(conversationId) as { id: string; title: string } | undefined;

      if (!conv) {
        throw new Error("Conversation does not exist.");
      }

      this.db
        .prepare("DELETE FROM messages WHERE conversation_id = ? AND in_context = 1")
        .run(conversationId);

      const insertStmt = this.db.prepare(
        `INSERT INTO messages (conversation_id, role, content, attachments, in_context, created_at)
         VALUES (?, ?, ?, ?, 1, ?)`
      );

      for (const message of messages) {
        if (message.role !== "user" && message.role !== "model") {
          throw new Error("Invalid message role.");
        }
        insertStmt.run(
          conversationId,
          message.role,
          message.text ?? "",
          JSON.stringify(message.attachments ?? []),
          now
        );
      }

      let title = conv.title;
      if (title === "New Chat") {
        const firstUser = messages.find((m) => m.role === "user");
        if (firstUser) {
          title = (firstUser.text ?? "").trim().slice(0, 50) || "New Chat";
        }
      }

      this.db
        .prepare(
          `UPDATE conversations
           SET title = ?, summary = ?, updated_at = ?
           WHERE id = ?`
        )
        .run(title, summary || "", now, conversationId);
    });
  }

  /**
   * Remove the last user+model exchange from a conversation (used for regenerate).
   * Returns the removed user prompt text and attachments if found, otherwise null.
   */
  popLastTurnWithAttachments(
    conversationId: string
  ): { text: string; attachments: MessageAttachment[] } | null {
    return this.runInTransaction(() => {
      const rows = this.db
        .prepare(
          `SELECT id, role, content, attachments FROM messages
           WHERE conversation_id = ?
           ORDER BY id DESC
           LIMIT 2`
        )
        .all(conversationId) as Array<{
          id: number;
          role: string;
          content: string;
          attachments: string;
        }>;

      if (rows.length === 0) return null;

      let result: { text: string; attachments: MessageAttachment[] } | null = null;
      const deleteStmt = this.db.prepare("DELETE FROM messages WHERE id = ?");

      if (rows[0].role === "model" && rows[1]?.role === "user") {
        result = {
          text: rows[1].content,
          attachments: Database.parseAttachments(rows[1].attachments),
        };
        deleteStmt.run(rows[0].id);
        deleteStmt.run(rows[1].id);
      } else if (rows[0].role === "user") {
        result = {
          text: rows[0].content,
          attachments: Database.parseAttachments(rows[0].attachments),
        };
        deleteStmt.run(rows[0].id);
      }

      if (result !== null) {
        this.db
          .prepare("UPDATE conversations SET updated_at = ? WHERE id = ?")
          .run(Database.now(), conversationId);
      }

      return result;
    });
  }

  /**
   * Remove the last user+model exchange from a conversation (used for regenerate).
   * Returns the removed user prompt text if found, otherwise null.
   */
  popLastTurn(conversationId: string): string | null {
    return this.popLastTurnWithAttachments(conversationId)?.text ?? null;
  }

  /** Returns full display transcript (archived + active messages). */
  getMessages(conversationId: string): StoredMessage[] {
    const rows = this.db
      .prepare(
        `SELECT id, role, content, attachments, in_context, created_at
         FROM messages
         WHERE conversation_id = ?
         ORDER BY id ASC`
      )
      .all(conversationId) as unknown as RawMessageRow[];

    return rows.map((row) => ({
      id: row.id,
      role: row.role as MessageRole,
      content: row.content,
      attachments: Database.parseAttachments(row.attachments),
      in_context: row.in_context === 1,
      created_at: row.created_at,
    }));
  }

  /** Returns only messages currently active in the LLM context window. */
  getContextMessages(conversationId: string): StoredMessage[] {
    const rows = this.db
      .prepare(
        `SELECT id, role, content, attachments, in_context, created_at
         FROM messages
         WHERE conversation_id = ? AND in_context = 1
         ORDER BY id ASC`
      )
      .all(conversationId) as unknown as RawMessageRow[];

    return rows.map((row) => ({
      id: row.id,
      role: row.role as MessageRole,
      content: row.content,
      attachments: Database.parseAttachments(row.attachments),
      in_context: true,
      created_at: row.created_at,
    }));
  }

  getConversation(conversationId: string): ConversationRecord | null {
    const row = this.db
      .prepare(
        `SELECT id, title, summary, custom_instructions, personality, response_length, response_tone, created_at, updated_at
         FROM conversations
         WHERE id = ?`
      )
      .get(conversationId) as unknown as RawConversationRow | undefined;

    if (!row) return null;
    return {
      id: row.id,
      title: row.title,
      summary: row.summary,
      custom_instructions: row.custom_instructions || "",
      personality: (row.personality || "") as PersonalityId | "",
      response_length: (row.response_length || "") as ResponseLength | "",
      response_tone: (row.response_tone || "") as ResponseTone | "",
      created_at: row.created_at,
      updated_at: row.updated_at,
    };
  }

  updateConversationSettings(
    conversationId: string,
    patch: {
      custom_instructions?: string;
      personality?: PersonalityId | "";
      response_length?: ResponseLength | "";
      response_tone?: ResponseTone | "";
    }
  ): ConversationRecord | null {
    const existing = this.getConversation(conversationId);
    if (!existing) return null;

    const customInstructions =
      patch.custom_instructions !== undefined
        ? patch.custom_instructions.trim().slice(0, 1000)
        : existing.custom_instructions;
    const personality =
      patch.personality !== undefined ? patch.personality : existing.personality;
    const responseLength =
      patch.response_length !== undefined
        ? patch.response_length
        : existing.response_length;
    const responseTone =
      patch.response_tone !== undefined
        ? patch.response_tone
        : existing.response_tone;

    const now = Database.now();
    this.db
      .prepare(
        `UPDATE conversations
         SET custom_instructions = ?, personality = ?, response_length = ?, response_tone = ?, updated_at = ?
         WHERE id = ?`
      )
      .run(
        customInstructions,
        personality,
        responseLength,
        responseTone,
        now,
        conversationId
      );

    return this.getConversation(conversationId);
  }

  renameConversation(conversationId: string, title: string): boolean {
    const cleaned = title.trim().slice(0, 80);
    if (!cleaned) return false;
    const info = this.db
      .prepare("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?")
      .run(cleaned, Database.now(), conversationId);
    return info.changes > 0;
  }

  updateSummary(conversationId: string, summary: string): boolean {
    const info = this.db
      .prepare("UPDATE conversations SET summary = ?, updated_at = ? WHERE id = ?")
      .run(summary, Database.now(), conversationId);
    return info.changes > 0;
  }

  getConversations(searchQuery?: string): ConversationSummaryItem[] {
    const q = (searchQuery ?? "").trim();

    if (q) {
      const pattern = `%${q.replace(/[\\%_]/g, "\\$&")}%`;
      const rows = this.db
        .prepare(
          `SELECT DISTINCT
             c.id,
             c.title,
             c.summary,
             c.custom_instructions,
             c.created_at,
             c.updated_at,
             (SELECT COUNT(*) FROM messages m2 WHERE m2.conversation_id = c.id) AS message_count
           FROM conversations c
           LEFT JOIN messages m ON m.conversation_id = c.id
           WHERE c.title LIKE ? ESCAPE '\\' OR m.content LIKE ? ESCAPE '\\'
           ORDER BY c.updated_at DESC`
        )
        .all(pattern, pattern) as unknown as Array<
        RawConversationRow & { message_count: number }
      >;

      return rows.map((row) => ({
        id: row.id,
        title: row.title,
        has_summary: Boolean(row.summary),
        has_custom_instructions: Boolean(row.custom_instructions),
        message_count: Number(row.message_count) || 0,
        created_at: row.created_at,
        updated_at: row.updated_at,
      }));
    }

    const rows = this.db
      .prepare(
        `SELECT
           c.id,
           c.title,
           c.summary,
           c.custom_instructions,
           c.created_at,
           c.updated_at,
           (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
         FROM conversations c
         ORDER BY c.updated_at DESC`
      )
      .all() as unknown as Array<RawConversationRow & { message_count: number }>;

    return rows.map((row) => ({
      id: row.id,
      title: row.title,
      has_summary: Boolean(row.summary),
      has_custom_instructions: Boolean(row.custom_instructions),
      message_count: Number(row.message_count) || 0,
      created_at: row.created_at,
      updated_at: row.updated_at,
    }));
  }

  deleteConversation(conversationId: string): boolean {
    const info = this.db
      .prepare("DELETE FROM conversations WHERE id = ?")
      .run(conversationId);
    return info.changes > 0;
  }

  static defaultPreferences(): UserPreferences {
    return {
      consent_given: false,
      preferred_name: "",
      personality: "friendly",
      custom_system_prompt: "",
      response_length: "balanced",
      response_tone: "neutral",
      memory_enabled: false,
      model: "",
      updated_at: Database.now(),
    };
  }

  getPreferences(): UserPreferences {
    const row = this.db
      .prepare("SELECT * FROM user_preferences WHERE id = 'default'")
      .get() as unknown as RawPreferencesRow | undefined;

    if (!row || row.consent_given !== 1) {
      return Database.defaultPreferences();
    }

    return {
      consent_given: row.consent_given === 1,
      preferred_name: row.preferred_name || "",
      personality: (row.personality || "friendly") as PersonalityId,
      custom_system_prompt: row.custom_system_prompt || "",
      response_length: (row.response_length || "balanced") as ResponseLength,
      response_tone: (row.response_tone || "neutral") as ResponseTone,
      memory_enabled: row.memory_enabled === 1,
      model: row.model || "",
      updated_at: row.updated_at,
    };
  }

  savePreferences(patch: Partial<UserPreferences>): UserPreferences {
    const now = Database.now();
    const current = this.getPreferences();
    const consent =
      patch.consent_given !== undefined
        ? patch.consent_given
        : current.consent_given;

    if (!consent) {
      this.db.prepare("DELETE FROM user_preferences WHERE id = 'default'").run();
      return Database.defaultPreferences();
    }

    const next: UserPreferences = {
      consent_given: true,
      preferred_name:
        patch.preferred_name !== undefined
          ? patch.preferred_name.trim().slice(0, 40)
          : current.preferred_name,
      personality: patch.personality ?? current.personality,
      custom_system_prompt:
        patch.custom_system_prompt !== undefined
          ? patch.custom_system_prompt.trim().slice(0, 1500)
          : current.custom_system_prompt,
      response_length: patch.response_length ?? current.response_length,
      response_tone: patch.response_tone ?? current.response_tone,
      memory_enabled:
        patch.memory_enabled !== undefined
          ? patch.memory_enabled
          : current.memory_enabled,
      model:
        patch.model !== undefined ? patch.model.trim().slice(0, 64) : current.model,
      updated_at: now,
    };

    this.db
      .prepare(
        `INSERT INTO user_preferences (
           id, consent_given, preferred_name, personality, custom_system_prompt,
           response_length, response_tone, memory_enabled, model, updated_at
         ) VALUES ('default', ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           consent_given = excluded.consent_given,
           preferred_name = excluded.preferred_name,
           personality = excluded.personality,
           custom_system_prompt = excluded.custom_system_prompt,
           response_length = excluded.response_length,
           response_tone = excluded.response_tone,
           memory_enabled = excluded.memory_enabled,
           model = excluded.model,
           updated_at = excluded.updated_at`
      )
      .run(
        1,
        next.preferred_name,
        next.personality,
        next.custom_system_prompt,
        next.response_length,
        next.response_tone,
        next.memory_enabled ? 1 : 0,
        next.model,
        now
      );

    return next;
  }

  resetPreferences(): UserPreferences {
    this.db.prepare("DELETE FROM user_preferences WHERE id = 'default'").run();
    return Database.defaultPreferences();
  }

  getMemories(): MemoryRecord[] {
    const rows = this.db
      .prepare(
        `SELECT id, content, category, source_conversation_id, created_at, updated_at
         FROM memories
         ORDER BY updated_at DESC, id DESC`
      )
      .all() as unknown as RawMemoryRow[];

    return rows.map((r) => ({
      id: r.id,
      content: r.content,
      category: (r.category || "general") as MemoryCategory,
      source_conversation_id: r.source_conversation_id ?? null,
      created_at: r.created_at,
      updated_at: r.updated_at,
    }));
  }

  addMemory(
    content: string,
    category: MemoryCategory = "general",
    sourceConversationId: string | null = null
  ): MemoryRecord {
    const cleaned = content.trim().replace(/\s+/g, " ").slice(0, 300);
    if (!cleaned) {
      throw new Error("Memory content cannot be empty.");
    }
    if (isSensitiveMemoryContent(cleaned)) {
      throw new Error(
        "Sensitive information such as API keys, passwords, tokens, or ID numbers cannot be saved to long-term memory."
      );
    }

    const existing = this.db
      .prepare("SELECT * FROM memories WHERE LOWER(content) = LOWER(?) LIMIT 1")
      .get(cleaned) as unknown as RawMemoryRow | undefined;

    const now = Database.now();
    if (existing) {
      this.db
        .prepare("UPDATE memories SET category = ?, updated_at = ? WHERE id = ?")
        .run(category, now, existing.id);
      return {
        id: existing.id,
        content: existing.content,
        category,
        source_conversation_id: existing.source_conversation_id ?? null,
        created_at: existing.created_at,
        updated_at: now,
      };
    }

    const info = this.db
      .prepare(
        `INSERT INTO memories (content, category, source_conversation_id, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?)`
      )
      .run(cleaned, category, sourceConversationId, now, now);

    return {
      id: Number(info.lastInsertRowid),
      content: cleaned,
      category,
      source_conversation_id: sourceConversationId,
      created_at: now,
      updated_at: now,
    };
  }

  deleteMemory(memoryId: number): boolean {
    const info = this.db
      .prepare("DELETE FROM memories WHERE id = ?")
      .run(memoryId);
    return info.changes > 0;
  }

  clearMemories(): number {
    const info = this.db.prepare("DELETE FROM memories").run();
    return Number(info.changes) || 0;
  }

  clearDatabase(): void {
    this.db.exec("DELETE FROM conversations;");
    this.db.exec("DELETE FROM user_preferences;");
    this.db.exec("DELETE FROM memories;");
  }

  close(): void {
    try {
      if (this.databasePath !== ":memory:") {
        this.db.exec("PRAGMA wal_checkpoint(TRUNCATE);");
      }
    } catch {
      // Ignore checkpoint error during shutdown
    }
    this.db.close();
  }
}
