
"""
ConvoAI - FastAPI Backend

Run: uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs
UI:   http://127.0.0.1:8000/
"""

import logging
import os

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

log = logging.getLogger("convoai")

from chat_engine import MAX_MESSAGE_CHARS, ChatEngine
from database import Database
from errors import ChatError

STATIC_DIR = Path(__file__).parent / "static"


# ------------------------------------------
# REQUEST / RESPONSE MODELS
# ------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        max_length=MAX_MESSAGE_CHARS
    )

    conversation_id: str | None = Field(
        default=None,
        max_length=64
    )

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Message cannot be blank.")

        return value


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    messages_in_context: int
    has_summary: bool


class ResetRequest(BaseModel):
    conversation_id: str | None = Field(
        default=None,
        max_length=64
    )


# ------------------------------------------
# APPLICATION FACTORY
# ------------------------------------------

def create_app(
    engine: ChatEngine | None = None,
    database: Database | None = None
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        # Initialize SQLite.
        if app.state.database is None:
            app.state.database = Database()

        # Initialize Gemini-backed ChatEngine.
        if app.state.engine is None:
            from llm_client import build_client

            llm = build_client()

            app.state.engine = ChatEngine(
                llm,
                database=app.state.database
            )

        yield

    app = FastAPI(
        title="ConvoAI",
        description="AI chatbot powered by Google Gemini",
        version="1.1.0",
        lifespan=lifespan
    )

    app.state.engine = engine
    app.state.database = database

    # --------------------------------------
    # CORS
    # --------------------------------------

    origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]

    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"]
        )

    # --------------------------------------
    # ERROR HANDLERS
    # --------------------------------------

    @app.exception_handler(ChatError)
    async def chat_error_handler(request: Request, exc: ChatError):
        return JSONResponse(
            {"error": exc.user_message},
            status_code=exc.status
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            {
                "error": (
                    f"Invalid request: message must contain "
                    f"1-{MAX_MESSAGE_CHARS} characters."
                )
            },
            status_code=400
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        log.exception("Unexpected error on %s", request.url.path)

        return JSONResponse(
            {"error": "Unexpected server error. Please try again."},
            status_code=500
        )

    # --------------------------------------
    # DEPENDENCIES
    # --------------------------------------

    def get_engine(request: Request) -> ChatEngine:
        return request.app.state.engine

    def get_database(request: Request) -> Database:
        return request.app.state.database

    # --------------------------------------
    # FRONTEND
    # --------------------------------------

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC_DIR / "index.html")

    # --------------------------------------
    # BASIC ENDPOINTS
    # --------------------------------------

    @app.get("/api", tags=["meta"])
    def api_info():
        return {
            "message": "Welcome to ConvoAI!",
            "status": "Backend is running",
            "docs": "/docs"
        }

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "healthy"}

    # --------------------------------------
    # CHAT ENDPOINT
    # --------------------------------------

    @app.post(
        "/api/chat",
        response_model=ChatResponse,
        tags=["chat"]
    )
    def chat(
        req: ChatRequest,
        engine: ChatEngine = Depends(get_engine)
    ):
        return engine.send(
            req.conversation_id,
            req.message
        )

    # --------------------------------------
    # RESET ENDPOINT
    # --------------------------------------

    @app.post("/api/reset", tags=["chat"])
    def reset(
        req: ResetRequest,
        engine: ChatEngine = Depends(get_engine)
    ):
        engine.reset(req.conversation_id)

        return {"status": "reset"}

    # --------------------------------------
    # LIST CONVERSATIONS
    # --------------------------------------

    @app.get("/api/conversations", tags=["conversations"])
    def list_conversations(
        database: Database = Depends(get_database)
    ):
        conversations = database.get_conversations()

        return {
            "conversations": conversations,
            "total": len(conversations)
        }

    # --------------------------------------
    # GET CONVERSATION DETAILS
    # --------------------------------------

    @app.get(
        "/api/conversations/{conversation_id}",
        tags=["conversations"]
    )
    def get_conversation(
        conversation_id: str,
        database: Database = Depends(get_database)
    ):
        conversation = database.get_conversation(conversation_id)

        if conversation is None:
            return JSONResponse(
                {"error": "Conversation not found."},
                status_code=404
            )

        messages = database.get_messages(conversation_id)

        return {
            "conversation": conversation,
            "messages": messages
        }

    # --------------------------------------
    # DELETE CONVERSATION
    # --------------------------------------

    @app.delete(
        "/api/conversations/{conversation_id}",
        tags=["conversations"]
    )
    def delete_conversation(
        conversation_id: str,
        engine: ChatEngine = Depends(get_engine),
        database: Database = Depends(get_database)
    ):
        # Remove it from the in-memory cache.
        engine.reset(conversation_id)

        # Also remove it from SQLite.
        deleted = database.delete_conversation(conversation_id)

        if not deleted:
            return JSONResponse(
                {"error": "Conversation not found."},
                status_code=404
            )

        return {
            "status": "deleted",
            "conversation_id": conversation_id
        }

    return app


# ------------------------------------------
# APPLICATION INSTANCE
# ------------------------------------------

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
