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

            # If the paragraph still fits in the current chunk
            if (
                len(current_chunk) + len(paragraph)
                <= chunk_size
            ):
                current_chunk += paragraph + "\n"

            # If it would exceed the limit,
            # close the current chunk
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
You are analyzing ONE SINGLE candidate.

Use only the information explicitly present
in the resume provided below.

Rules:

- Do not invent technologies.
- Do not invent experience.
- Do not infer knowledge that is not explicitly stated.
- Extract only information relevant to the question.
- Do not turn front-end experience into back-end experience.
- Do not consider Docker, Git or CI/CD as evidence of back-end experience.
- If a technology is not explicitly mentioned in the context,
  do not include it.
- If no back-end technologies are explicitly mentioned,
  return backend_technologies as an empty list.
- If no back-end experience is explicitly mentioned,
  return backend_experiences as an empty list.

RESUME:
{context}

QUESTION:
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
    Answer the user's question using only
    the structured candidate analysis provided below.

    Do not invent information.
    Do not add technologies or experience that are not
    present in the analysis.
    If there is not enough information, state that explicitly.

    QUESTION:
    {question}

    CANDIDATE ANALYSIS:
    {analysis.model_dump_json(indent=2)}

    Provide a direct and objective answer.
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
