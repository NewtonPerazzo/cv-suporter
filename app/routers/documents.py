from fastapi import APIRouter, File, UploadFile

from app.services.documents import documents_service


documents_router = APIRouter(prefix="/api/v1")

@documents_router.post("/documents", tags=["Documents"])
async def post_documents(
    data: list[UploadFile] = File(
        ...,
        description="List of files to upload",
        json_schema_extra={
            "items": {
                "type": "string",
                "format": "binary",
            }
        },
    ),
) -> dict:
    documents = await documents_service.post_documents(data)
    return { "message": documents }