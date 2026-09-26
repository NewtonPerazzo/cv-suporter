from fastapi import APIRouter, UploadFile, File

from app.models.langchain_documents import DocumentQuery
from app.services.langchain_documents import (
    langchain_documents_service
)


langchain_documents_router = APIRouter()


@langchain_documents_router.post("/langchain_documents")
async def upload_documents(
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
):
    document_ids = await langchain_documents_service.process_documents(
        data
    )

    return {
        "document_ids": document_ids
    }

@langchain_documents_router.post("/langchain_query")
def query_documents(
    query: DocumentQuery
):
    result = langchain_documents_service.query_documents(
        question=query.question,
        document_ids=query.document_ids
    )

    return {
        "answer": result["answer"],
        "documents": [
            {
                "text": document.page_content,
                "metadata": document.metadata
            }
            for document in result["documents"]
        ]
    }