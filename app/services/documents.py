from fastapi import File, UploadFile
import pymupdf


class DocumentsService:
    def __init__(self):
        pass

    async def post_documents(
        self, 
        documents: list[UploadFile] = File(..., description="List of files to upload")
    ) -> list[str]:
        documents_text_list = []
        chunk_documents_text_list = []

        for doc in documents:
            doc_item = await self.extract_pdf_text(doc)
            documents_text_list.append(doc_item)

        for doc_text in documents_text_list:
            chunks = self.chunk_text(doc_text)
            chunk_documents_text_list.extend(chunks)

        return chunk_documents_text_list

    async def extract_pdf_text(self, file: UploadFile) -> str:
        content = await file.read()
        pdf = pymupdf.open(
            stream=content,
            filetype="pdf"
        )
        text = ""

        for page in pdf:
            text += page.get_text()

        pdf.close()

        return text

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

documents_service = DocumentsService()