from typing import Annotated
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])
bearer = HTTPBearer()


def service(request: Request):
    return request.app.state.chat


def session_token(credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer)]):
    return credentials.credentials


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] = Field(min_length=1, max_length=10)


@router.post("/sessions", status_code=201)
def create_session(chat=Depends(service)):
    return chat.create_session()


@router.get("/session")
def session_status(token=Depends(session_token), chat=Depends(service)):
    return chat.status(token)


@router.delete("/session", status_code=204)
def end_session(token=Depends(session_token), chat=Depends(service)):
    chat.end_session(token)


@router.post("/documents")
async def upload_documents(
    files: Annotated[list[UploadFile], File(json_schema_extra={"items": {"type": "string", "format": "binary"}})],
    token=Depends(session_token), chat=Depends(service),
):
    try:
        await run_in_threadpool(chat.status, token)
        if not 1 <= len(files) <= 10:
            raise HTTPException(422, "Upload between 1 and 10 PDFs.")
        payload = []
        total = 0
        for file in files:
            content = await file.read(5 * 1024 * 1024 + 1)
            total += len(content)
            if len(content) > 5 * 1024 * 1024 or total > 20 * 1024 * 1024:
                raise HTTPException(413, "Limit: 5 MB per PDF and 20 MB per upload.")
            if not content.startswith(b"%PDF-"):
                raise HTTPException(422, "Only PDF files are supported.")
            filename = (file.filename or "resume.pdf").replace("\\", "/").split("/")[-1][:200]
            payload.append((filename, content))
        return await run_in_threadpool(chat.upload, token, payload)
    finally:
        for file in files:
            await file.close()


@router.post("/messages")
def send_message(body: Question, token=Depends(session_token), chat=Depends(service)):
    if not body.question.strip():
        raise HTTPException(422, "Enter a question.")
    return chat.query(token, body.question.strip(), body.document_ids)
