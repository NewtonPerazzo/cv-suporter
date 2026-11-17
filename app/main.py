import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from app.config.settings import get_settings
from app.routers.chat import router
from app.services.chat import ChatService

settings = get_settings()
logger = logging.getLogger(__name__)


async def cleanup_loop(chat):
    while True:
        await asyncio.sleep(settings.cleanup_interval_seconds)
        try:
            await run_in_threadpool(chat.cleanup)
        except Exception:
            logger.exception("Session cleanup failed; retrying on the next interval")


@asynccontextmanager
async def lifespan(app):
    chat = ChatService()
    await run_in_threadpool(chat.initialize)
    app.state.chat = chat
    cleanup_task = asyncio.create_task(cleanup_loop(chat))
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        chat.engine.dispose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip().rstrip("/") for origin in settings.cors_origins.split(",") if origin.strip()],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(router)

# Original exercises predate session isolation; enable only for local study.
if settings.enable_study_routes:
    from app.routers.documents import documents_router
    from app.routers.langchain_documents import langchain_documents_router
    app.include_router(documents_router)
    app.include_router(langchain_documents_router)


@app.middleware("http")
async def private_responses(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
def health():
    with app.state.chat.engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok"}
