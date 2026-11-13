import test from "node:test";
import assert from "node:assert/strict";
import { Database } from "../src/database.js";

test("Database stores, compacts, searches, renames, and deletes conversations in SQLite", () => {
  const db = new Database(":memory:");
  try {
    db.createConversation("conv-1");
    db.addMessage("conv-1", "user", "Hello ConvoAI, how do binary trees work?");
    db.addMessage("conv-1", "model", "A binary tree is a hierarchical data structure.");

    const conv = db.getConversation("conv-1");
    assert.ok(conv);
    assert.equal(conv.title, "Hello ConvoAI, how do binary trees work?");

    // Simulate compaction archiving the first 2 messages while adding turn 2
    db.saveTurnState(
      "conv-1",
      "Explain AVL balancing",
      "AVL trees maintain height balance via rotations.",
      2,
      "User asked about binary trees."
    );

    const allMessages = db.getMessages("conv-1");
    const activeContext = db.getContextMessages("conv-1");
    assert.equal(allMessages.length, 4);
    assert.equal(activeContext.length, 2);
    assert.equal(allMessages[0].in_context, false);
    assert.equal(allMessages[2].in_context, true);

    const updated = db.getConversation("conv-1");
    assert.equal(updated?.summary, "User asked about binary trees.");

    // Search matches message content even when title differs
    assert.equal(db.getConversations("rotations").length, 1);
    assert.equal(db.getConversations("nonexistent-term").length, 0);

    // Rename
    assert.equal(db.renameConversation("conv-1", "Data Structures Notes"), true);
    assert.equal(db.getConversation("conv-1")?.title, "Data Structures Notes");

    // Pop last turn
    const poppedPrompt = db.popLastTurn("conv-1");
    assert.equal(poppedPrompt, "Explain AVL balancing");
    assert.equal(db.getMessages("conv-1").length, 2);

    // Delete cascades
    assert.equal(db.deleteConversation("conv-1"), true);
    assert.equal(db.getConversation("conv-1"), null);
    assert.equal(db.getMessages("conv-1").length, 0);
  } finally {
    db.close();
  }
});

test("Database persists message attachments, pops turns with attachments, stores model preferences, and migrates legacy tables", async () => {
  const os = await import("node:os");
  const path = await import("node:path");
  const fs = await import("node:fs");
  const { DatabaseSync } = await import("node:sqlite");

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "convoai-db-mig-"));
  const dbPath = path.join(tmpDir, "legacy.db");

  // Create a legacy SQLite database without attachments/model columns
  const legacy = new DatabaseSync(dbPath);
  legacy.exec(`
    CREATE TABLE conversations (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL DEFAULT 'New Chat',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE TABLE messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      conversation_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE TABLE user_preferences (
      id TEXT PRIMARY KEY,
      consent_given INTEGER NOT NULL DEFAULT 0,
      preferred_name TEXT NOT NULL DEFAULT '',
      personality TEXT NOT NULL DEFAULT 'friendly',
      custom_system_prompt TEXT NOT NULL DEFAULT '',
      response_length TEXT NOT NULL DEFAULT 'balanced',
      response_tone TEXT NOT NULL DEFAULT 'neutral',
      memory_enabled INTEGER NOT NULL DEFAULT 0,
      updated_at TEXT NOT NULL
    );
  `);
  legacy.close();

  const db = new Database(dbPath);
  try {
    db.createConversation("att-conv");
    const attachments = [
      {
        kind: "image" as const,
        name: "diagram.png",
        mimeType: "image/png",
        data: "iVBORw0KGgo=",
      },
      {
        kind: "text" as const,
        name: "notes.md",
        mimeType: "text/markdown",
        textContent: "# Architecture\nUses SQLite.",
      },
      {
        kind: "pdf" as const,
        name: "spec.pdf",
        mimeType: "application/pdf",
        data: "JVBERi0xLjQ=",
      },
    ];

    db.saveTurnState(
      "att-conv",
      "",
      "Analyzed all three files.",
      0,
      "",
      attachments
    );

    const msgs = db.getMessages("att-conv");
    assert.equal(msgs.length, 2);
    assert.equal(msgs[0].attachments.length, 3);
    assert.equal(msgs[0].attachments[0].kind, "image");
    assert.equal(msgs[0].attachments[1].textContent, "# Architecture\nUses SQLite.");
    assert.equal(msgs[0].attachments[2].kind, "pdf");

    const conv = db.getConversation("att-conv");
    assert.equal(conv?.title, "Attachment: diagram.png");

    // Pop last turn preserving attachments
    const popped = db.popLastTurnWithAttachments("att-conv");
    assert.ok(popped);
    assert.equal(popped.text, "");
    assert.equal(popped.attachments.length, 3);
    assert.equal(db.getMessages("att-conv").length, 0);

    // Model preference saving & reset
    const savedPrefs = db.savePreferences({
      consent_given: true,
      model: "gemini-3.1-flash-lite",
    });
    assert.equal(savedPrefs.model, "gemini-3.1-flash-lite");
    assert.equal(db.getPreferences().model, "gemini-3.1-flash-lite");

    const resetPrefs = db.resetPreferences();
    assert.equal(resetPrefs.model, "");
    assert.equal(db.getPreferences().model, "");
  } finally {
    db.close();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

