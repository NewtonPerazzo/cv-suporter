from pydantic import BaseModel


class DocumentQuery(BaseModel):
    question: str
    document_ids: list[str]