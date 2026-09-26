from fastapi import UploadFile
from pydantic import BaseModel


class DocumentsCreate(BaseModel):
    file: UploadFile

class CandidateLLMAnalysis(BaseModel):
    backend_technologies: list[str]
    backend_experiences: list[str]

class CandidateAnalysis(BaseModel):
    filename: str
    backend_technologies: list[str]
    backend_experiences: list[str]


class CandidatesAnalysis(BaseModel):
    candidates: list[CandidateAnalysis]