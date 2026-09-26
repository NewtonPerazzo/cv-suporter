from uuid import uuid4

import pymupdf

from fastapi import UploadFile

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from app.config.settings import get_settings


settings = get_settings()


class LangchainDocumentsService:

    def __init__(self):

        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="candidate_cv",
            connection=settings.database_url,
            use_jsonb=True
        )

        self.llm = ChatOllama(
            model="qwen3:1.7b",
            temperature=0
        )

        self.answer_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
            You are an assistant responsible for analyzing resumes.

            Use only the information present in the provided context.

            Rules:
            - Do not invent experience, technologies or qualifications.
            - Do not assume knowledge that is not explicitly present.
            - Compare candidates only based on the retrieved evidence.
            - If there is not enough information, state that clearly.
            - Do not treat semantic similarity as a measure of experience or competence.
            """
                ),
                (
                    "human",
                    """
            CONTEXT:

            {context}

            QUESTION:

            {question}
            """
                )
            ])

    def query_documents(
        self,
        question: str,
        document_ids: list[str]
    ) -> dict:

        documents = self.retrieve_by_document(
            question=question,
            document_ids=document_ids,
            top_k_per_document=3
        )

        answer = self.generate_answer(
            question=question,
            documents=documents
        )

        return {
            "answer": answer,
            "documents": documents
        }

    def generate_answer(
        self,
        question: str,
        documents: list[Document]
    ) -> str:

        context = "\n\n---\n\n".join(
            f"""
    CANDIDATE: {document.metadata["filename"]}

    {document.page_content}
    """
            for document in documents
        )

        chain = self.answer_prompt | self.llm

        response = chain.invoke({
            "question": question,
            "context": context
        })

        return response.content

    async def process_documents(
        self,
        documents: list[UploadFile]
    ) -> list[str]:

        document_ids = []

        for document in documents:

            document_id = str(uuid4())

            text = await self.extract_pdf_text(
                document
            )

            chunks = self.split_document(
                text=text,
                filename=document.filename,
                document_id=document_id
            )

            self.add_documents(chunks)

            document_ids.append(document_id)

        return document_ids

    async def extract_pdf_text(
        self,
        file: UploadFile
    ) -> str:

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

    def split_document(
        self,
        text: str,
        filename: str,
        document_id: str
    ) -> list[Document]:

        document = Document(
            page_content=text,
            metadata={
                "document_id": document_id,
                "filename": filename
            }
        )

        chunks = self.text_splitter.split_documents(
            [document]
        )

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

        return chunks

    def add_documents(
        self,
        documents: list[Document]
    ) -> None:

        self.vector_store.add_documents(
            documents=documents
        )

    def retrieve(
        self,
        question: str,
        document_ids: list[str],
        top_k: int = 3
    ) -> list[Document]:

        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": top_k,
                "filter": {
                    "document_id": {
                        "$in": document_ids
                    }
                }
            }
        )

        return retriever.invoke(question)

    def retrieve_by_document(
        self,
        question: str,
        document_ids: list[str],
        top_k_per_document: int = 3
    ) -> list[Document]:

        results = []

        for document_id in document_ids:

            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "k": top_k_per_document,
                    "filter": {
                        "document_id": {
                            "$eq": document_id
                        }
                    }
                }
            )

            documents = retriever.invoke(question)

            results.extend(documents)

        return results


langchain_documents_service = LangchainDocumentsService()
