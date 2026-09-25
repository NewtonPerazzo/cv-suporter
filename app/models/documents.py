from fastapi import UploadFile
from pydantic import BaseModel


class DocumentsCreate(BaseModel):
    file: UploadFile