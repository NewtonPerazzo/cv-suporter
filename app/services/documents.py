from fastapi import File, UploadFile
import pymupdf
import ollama
import numpy as np

from app.models.documents import (
    CandidatesAnalysis,
    CandidateAnalysis,
    CandidateLLMAnalysis
)


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

        # 1. generate question embedding
        embedding_question = self.ollama_embedding_text(question)

        # 2. extract, divide and embed each document
        embedding_response = await self.process_documents(documents)

        results = []

        # 3. calculate similarity between question embedding and each chunk embedding
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

        # 4. take top_k_per_file chunks for each candidate
        sorted_results = self.sort_documents_by_similarity_and_file(
            documents=results,
            top_k_per_file=3
        )

        # 5. analyse each candidate individually and generate a final answer
        analysis = self.generate_answer(
            question=question,
            results=sorted_results
        )

        # 6. generate final answer based on the analysis of all candidates      
        answer = self.generate_final_answer(
            question=question,
            analysis=analysis
        )

        return {
            "analysis": analysis,
            "answer": answer,
            "sources": sorted_results
        }

    # ---------------------------------------------------------
    # RETRIEVAL
    # ---------------------------------------------------------

    def sort_documents_by_similarity_and_file(
        self,
        documents: list[dict],
        top_k_per_file: int = 3
    ) -> list[dict]:

        grouped = {}

        for doc in documents:
            filename = doc["filename"]

            if filename not in grouped:
                grouped[filename] = []

            grouped[filename].append(doc)

        top_results = []

        for docs in grouped.values():
            docs.sort(
                key=lambda x: x["similarity"],
                reverse=True
            )

            top_results.extend(
                docs[:top_k_per_file]
            )

        return top_results

    def group_by_candidate(
        self,
        results: list[dict]
    ) -> dict[str, list[dict]]:

        grouped = {}

        for result in results:
            filename = result["filename"]

            if filename not in grouped:
                grouped[filename] = []

            grouped[filename].append(result)

        return grouped

    # ---------------------------------------------------------
    # DOCUMENT PROCESSING
    # ---------------------------------------------------------

    async def process_documents(
        self,
        documents: list[UploadFile]
    ) -> list[dict]:

        chunk_documents_text_list = []

        for doc in documents:

            doc_text = await self.extract_pdf_text(doc)

            chunks = self.chunk_text_by_paragraph(
                doc_text
            )

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

    # ---------------------------------------------------------
    # PDF
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # CHUNKING
    # ---------------------------------------------------------

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

            # Se ainda cabe no chunk atual
            if (
                len(current_chunk) + len(paragraph)
                <= chunk_size
            ):
                current_chunk += paragraph + "\n"

            # Se ultrapassaria o limite,
            # fecha o chunk atual
            else:

                if current_chunk:
                    chunks.append(
                        current_chunk.strip()
                    )

                current_chunk = paragraph + "\n"

        if current_chunk:
            chunks.append(
                current_chunk.strip()
            )

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

    # ---------------------------------------------------------
    # EMBEDDINGS
    # ---------------------------------------------------------

    def ollama_embedding_text(
        self,
        text: str
    ) -> list[float]:

        response = ollama.embed(
            model="nomic-embed-text",
            input=text
        )

        return response["embeddings"][0]

    # ---------------------------------------------------------
    # SIMILARITY
    # ---------------------------------------------------------

    def cosine_similarity(
        self,
        vector_a: list[float],
        vector_b: list[float]
    ) -> float:

        dot_product = np.dot(
            vector_a,
            vector_b
        )

        norm_a = np.linalg.norm(
            vector_a
        )

        norm_b = np.linalg.norm(
            vector_b
        )

        similarity = dot_product / (
            norm_a * norm_b
        )

        return float(similarity)

    # ---------------------------------------------------------
    # GENERATION
    # ---------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        results: list[dict]
    ) -> CandidatesAnalysis:

        grouped = self.group_by_candidate(
            results
        )

        candidates_analysis = []

        for filename, candidate_results in grouped.items():

            candidate_analysis = self.analyze_candidate(
                question=question,
                filename=filename,
                results=candidate_results
            )

            candidates_analysis.append(
                candidate_analysis
            )

        return CandidatesAnalysis(
            candidates=candidates_analysis
        )

    def analyze_candidate(
        self,
        question: str,
        filename: str,
        results: list[dict]
    ) -> CandidateAnalysis:

        context = ""

        for result in results:

            context += f"""
{result["text"]}

---
"""

        prompt = f"""
Você está analisando UM ÚNICO candidato.

Utilize exclusivamente as informações explicitamente presentes
no currículo fornecido abaixo.

Regras:

- Não invente tecnologias.
- Não invente experiências.
- Não deduza conhecimentos que não estejam explicitamente escritos.
- Extraia apenas informações relevantes para a pergunta.
- Não transforme experiência front-end em experiência back-end.
- Não considere Docker, Git ou CI/CD como evidência de experiência back-end.
- Se uma tecnologia não estiver explicitamente mencionada no contexto,
  não a inclua.
- Se não houver tecnologias back-end explicitamente mencionadas,
  retorne backend_technologies como lista vazia.
- Se não houver experiências back-end explicitamente mencionadas,
  retorne backend_experiences como lista vazia.

CURRÍCULO:
{context}

PERGUNTA:
{question}
"""

        response = ollama.chat(
            model="qwen3:1.7b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format=CandidateLLMAnalysis.model_json_schema()
        )

        llm_analysis = CandidateLLMAnalysis.model_validate_json(
            response["message"]["content"]
        )

        return CandidateAnalysis(
            filename=filename,
            backend_technologies=(
                llm_analysis.backend_technologies
            ),
            backend_experiences=(
                llm_analysis.backend_experiences
            )
        )

    def generate_final_answer(
        self,
        question: str,
        analysis: CandidatesAnalysis
    ) -> str:

        prompt = f"""
    Responda à pergunta do usuário utilizando exclusivamente
    a análise estruturada dos candidatos fornecida abaixo.

    Não invente informações.
    Não adicione tecnologias ou experiências que não estejam
    presentes na análise.
    Se não houver informação suficiente, diga isso explicitamente.

    PERGUNTA:
    {question}

    ANÁLISE DOS CANDIDATOS:
    {analysis.model_dump_json(indent=2)}

    Forneça uma resposta direta e objetiva.
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