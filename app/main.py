from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.documents import documents_router
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


def parse_cors_origins(origins: str) -> list[str]:
    normalized_origins = [
        origin.strip().rstrip("/")
        for origin in origins.split(",")
        if origin.strip()
    ]
    return list(dict.fromkeys([*normalized_origins, *REQUIRED_CORS_ORIGINS]))

app = FastAPI(
    title=settings.app_name,
)

# cors_origins = parse_cors_origins(settings.BACKEND_CORS_ORIGINS)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=cors_origins,
#     allow_credentials="*" not in cors_origins,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["X-Total-Count"],
# )

app.include_router(
    documents_router
)


@app.get("/health")
def health():
    return {"status": "ok"}
