from fastapi import File, UploadFile
import pymupdf
import ollama
import numpy as np


class DocumentsService:
    def __init__(self):
        pass

    async def post_documents(
        self,
        question: str,
        documents: list[UploadFile] = File(
            ...,
            description="List of files to upload"
        )
    ) -> dict:
        embedding_question = self.ollama_embedding_text(question)
        embedding_response = await self.process_documents(documents)

        results = []

        for chunk in embedding_response:
            similarity = self.cosine_similarity(
                embedding_question,
                chunk["embedding"]
            )

            results.append({
                "filename": chunk["filename"],
                "text": chunk["text"],
                "similarity": similarity
            })

        results.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        top_results = results[:5]

        answer = self.generate_answer(
            question=question,
            results=top_results
        )

        return {
            "answer": answer,
            "sources": top_results
        }

    async def process_documents(
        self,
        documents: list[UploadFile]
    ) -> list[dict]:
        chunk_documents_text_list = []

        for doc in documents:
            doc_text = await self.extract_pdf_text(doc)
            chunks = self.chunk_text_by_paragraph(doc_text)
           
            for chunk in chunks:
                chunk_documents_text_list.append({
                    "filename": doc.filename,
                    "text": chunk
                })

        texts = [
            chunk["text"]
            for chunk in chunk_documents_text_list
        ]
        response = ollama.embed(
            model="nomic-embed-text",
            input=texts
        )
        embeddings = response["embeddings"]
        embeddings_response = []

        for chunk, embedding in zip(
            chunk_documents_text_list,
            embeddings
        ):
            embeddings_response.append({
                "filename": chunk["filename"],
                "text": chunk["text"],
                "embedding": embedding,
                "size": len(embedding)
            })

        return embeddings_response

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

    def chunk_text_by_paragraph(
        self,
        text: str,
        chunk_size: int = 1000
    ) -> list[str]:

        paragraphs = text.split("\n")

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # if it don't exceed the chunk size, add it to the current chunk
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + "\n"

            # close current chunk if it exceeds the chunk size
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                current_chunk = paragraph + "\n"

        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_text_by_character(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> list[str]:

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk = text[start:end]

            chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    def ollama_embedding_text(
        self,
        text: str
    ) -> list[float]:

        response = ollama.embed(
            model="nomic-embed-text",
            input=text
        )

        return response["embeddings"][0]

    def cosine_similarity(
        self,
        vector_a: list[float],
        vector_b: list[float]
    ) -> float:

        dot_product = np.dot(
            vector_a,
            vector_b
        )

        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        similarity = dot_product / (
            norm_a * norm_b
        )

        return float(similarity)

    def generate_answer(
        self,
        question: str,
        results: list[dict]
    ) -> str:

        context = ""

        for result in results:
            context += f"""
Arquivo: {result["filename"]}

Conteúdo:
{result["text"]}

---
"""

        prompt = f"""
Você é um assistente que responde perguntas sobre currículos.

Responda utilizando EXCLUSIVAMENTE informações explicitamente presentes
no contexto fornecido.

Regras:
- Não invente experiências, habilidades ou qualificações.
- Não assuma que o candidato possui uma habilidade que não esteja explicitamente mencionada.
- Não use conhecimento externo.
- Ao comparar candidatos, explique quais evidências do contexto sustentam a comparação.
- Caso o contexto recuperado não seja suficiente para comparar os candidatos,
  informe que não há informações suficientes.
- Não considere a pontuação de similaridade como medida de experiência ou competência.

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA:
"""

        response = ollama.chat(
            model="qwen3:1.7b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]


documents_service = DocumentsService()