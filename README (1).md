::: {align="center"}
# ConvoAI

### Conversations that remember. Intelligence that feels personal.

An AI-powered conversational assistant built with **Google Gemini**,
**FastAPI**, **Python**, and **SQLite**.

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Google
Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

[Overview](#-overview) · [Features](#-features) ·
[Architecture](#-architecture) · [Getting Started](#-getting-started) ·
[API](#-api-endpoints) · [Testing](#-testing)
:::

------------------------------------------------------------------------

## ✨ Overview

**ConvoAI** is a full-stack AI chat application designed to make
conversations with a large language model feel continuous and reliable.

Powered by Google's Gemini API, ConvoAI maintains conversation context
across multiple turns, stores conversation history in SQLite, and
manages long conversations through context compaction. When
summarization is unavailable, the chat engine can fall back to retaining
a recent portion of the conversation.

The project combines a Python backend with a browser-based chat
interface and is built as a practical implementation of real-world LLM
application concepts.

## 🚀 Features

-   **Gemini-powered responses** --- Connects to Google's Gemini model
    through the Google Gen AI SDK.
-   **Multi-turn conversations** --- Sends conversation history so
    replies can use earlier messages as context.
-   **Persistent chat history** --- Stores conversations and messages in
    SQLite.
-   **Conversation restoration** --- Loads saved conversation context
    when an existing conversation is continued.
-   **Context window management** --- Keeps the active conversation
    within a configurable history limit.
-   **Conversation summarization** --- Summarizes older messages to
    preserve useful context while retaining recent turns.
-   **Graceful fallback** --- Uses recent-message truncation if
    summarization fails.
-   **Error handling** --- Provides a structured application error path
    for failures.
-   **Configurable settings** --- Reads model and context settings from
    environment variables.
-   **Responsive chat interface** --- A browser-based interface with a
    dark, black-and-orange visual identity and a theme toggle.

## 🧱 Architecture

ConvoAI separates the API layer, conversation orchestration, model
integration, and persistence.

``` text
Browser
  │
  ▼
FastAPI application (main.py)
  │
  ▼
Chat engine (chat_engine.py)
  ├── Context management
  ├── Conversation history
  ├── Summarization / truncation
  │
  ├──────────────► Gemini client (llm_client.py)
  │                    │
  │                    ▼
  │               Google Gemini API
  │
  ▼
SQLite database (database.py)
  ├── Conversations
  └── Messages
```

### Project structure

``` text
ConvoAI/
├── static/
│   └── index.html          # Browser-based chat interface
├── main.py                 # FastAPI app and HTTP endpoints
├── chat_engine.py          # Conversation orchestration and context handling
├── llm_client.py           # Gemini client integration
├── database.py             # SQLite persistence layer
├── errors.py               # Application error definitions
├── test_gemini.py          # Gemini connectivity test
├── test_chat_engine.py     # Multi-turn conversation tests
├── test_context.py         # Context management tests
├── test_fallback.py        # Summarization fallback tests
├── test_database.py        # Database tests
├── test_persistence.py     # Persistence and restoration tests
├── .env.example            # Example environment configuration
├── .gitignore
└── README.md
```

## 🛠️ Tech Stack

  Layer                Technology              Purpose
  -------------------- ----------------------- ---------------------------------------
  Language             Python                  Backend implementation
  API framework        FastAPI                 HTTP API and application server
  LLM                  Google Gemini           Natural-language generation
  SDK                  Google Gen AI SDK       Gemini API communication
  Database             SQLite                  Persistent conversations and messages
  Configuration        python-dotenv           Load local environment variables
  Frontend             HTML, CSS, JavaScript   Chat user interface
  Development server   Uvicorn                 Run the FastAPI application

## ⚙️ Getting Started

### Prerequisites

-   Python 3.10 or newer (the project was developed and tested with
    Python 3.13).
-   A Google Gemini API key.
-   Git.

### 1. Clone the repository

``` bash
git clone https://github.com/pavankammila/ConvoAI.git
cd ConvoAI
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install fastapi uvicorn google-genai python-dotenv
```

### 4. Configure environment variables

Copy `.env.example` to `.env`.

**Windows PowerShell**

``` powershell
Copy-Item .env.example .env
```

**macOS / Linux**

``` bash
cp .env.example .env
```

Open `.env` and add your own Gemini API key and configuration:

``` env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_model_name
MAX_HISTORY=12
KEEP_RECENT=6
REQUEST_TIMEOUT=30
```

> **Security:** Never commit `.env` or share your API key. The
> repository's `.gitignore` excludes local environment files and SQLite
> database files. Keep `.env.example` free of real credentials.

### 5. Run the application

``` bash
python -m uvicorn main:app --reload
```

Open the application in your browser:

-   **Chat interface:** http://127.0.0.1:8000
-   **API documentation:** http://127.0.0.1:8000/docs
-   **Health endpoint:** http://127.0.0.1:8000/health

The SQLite database is initialized by the application and stores
conversation data locally.

## 🔌 API Endpoints

The FastAPI application exposes the following routes:

  ----------------------------------------------------------------------------------------
  Method                  Endpoint                                 Purpose
  ----------------------- ---------------------------------------- -----------------------
  `GET`                   `/`                                      Serve the chat
                                                                   interface

  `GET`                   `/api`                                   API information

  `GET`                   `/health`                                Health check

  `POST`                  `/api/chat`                              Send a message to a
                                                                   conversation

  `POST`                  `/api/reset`                             Reset conversation
                                                                   state

  `GET`                   `/api/conversations`                     List saved
                                                                   conversations

  `GET`                   `/api/conversations/{conversation_id}`   Retrieve a saved
                                                                   conversation

  `DELETE`                `/api/conversations/{conversation_id}`   Delete a conversation
  ----------------------------------------------------------------------------------------

Interactive request and response schemas are available at `/docs` while
the server is running.

## 🧠 How Conversation Memory Works

ConvoAI maintains a separate message history for each conversation.

1.  A user sends a message with a conversation ID.
2.  The chat engine loads or creates the corresponding conversation.
3.  The new message is added to the active history.
4.  If the history exceeds the configured limit, older messages are
    compacted.
5.  The engine attempts to summarize older context and retains recent
    messages.
6.  If summarization fails, the engine falls back to retaining recent
    messages.
7.  The Gemini client generates a response using the available context.
8.  The updated conversation state is saved to SQLite.

This approach helps balance continuity, context size, and resilience.

## 🧪 Testing

The project includes tests for its core behavior. Run them from the
project root with the virtual environment activated:

``` bash
python -m py_compile main.py chat_engine.py database.py llm_client.py errors.py

python test_database.py
python test_chat_engine.py
python test_context.py
python test_fallback.py
python test_persistence.py
```

To test connectivity with the live Gemini API:

``` bash
python test_gemini.py
```

The live API test requires a valid API key and an available model quota.
Provider-side rate limits or temporary service errors can affect this
test independently of the local unit tests.

## 🔐 Security Notes

-   Store credentials in `.env`, not in source code.
-   Do not commit `.env`, API keys, local databases, or virtual
    environments.
-   Use `.env.example` to document required configuration without
    including secrets.
-   Before deploying publicly, configure secrets in the hosting
    provider's environment settings.
-   Review CORS, authentication, rate limiting, and data-retention
    requirements before exposing the service to public users.

## 🗺️ Roadmap

Potential next steps for the project:

-   [ ] Deploy the application to a cloud hosting platform.
-   [ ] Add user authentication and per-user conversation ownership.
-   [ ] Add streaming responses.
-   [ ] Improve conversation search and organization.
-   [ ] Add configurable assistant personalities.
-   [ ] Add automated API and frontend integration tests.
-   [ ] Add usage limits and rate limiting for public deployments.

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome.

1.  Fork the repository.
2.  Create a feature branch: `git checkout -b feature/your-feature`.
3.  Commit your changes: `git commit -m "Add your feature"`.
4.  Push the branch: `git push origin feature/your-feature`.
5.  Open a pull request.

## 👨‍💻 Author

**Pavan Kumar**

B.Tech Computer Science & Engineering --- AI & ML

-   GitHub: [@pavankammila](https://github.com/pavankammila)

## 📄 License

No license has been added to this repository yet. Until a license is
selected, reuse and redistribution are not automatically permitted. Add
a license file if you want to grant others explicit permissions.

------------------------------------------------------------------------

::: {align="center"}
**Built with curiosity, code, and Gemini.**

⭐ If you find ConvoAI interesting, consider starring the repository.
:::
