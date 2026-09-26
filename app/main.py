from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.documents import documents_router
from app.routers.langchain_documents import langchain_documents_router
from app.config.settings import Settings

settings = Settings()

REQUIRED_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "https://perazzo-manager.vercel.app",
    "https://perazzo-catalog.vercel.app",
    "http://perazzo-catalog.vercel.app",
)

app = FastAPI(
    title=settings.app_name,
)

app.include_router(documents_router)
app.include_router(langchain_documents_router)


@app.get("/health")
def health():
    return {"status": "ok"}
