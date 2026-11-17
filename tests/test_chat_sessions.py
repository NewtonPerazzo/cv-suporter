"""Integration tests use an isolated PostgreSQL database with pgvector.

Set TEST_DATABASE_URL to a disposable database; never point at production.
Run: python -m unittest discover -s tests -v
"""
import os
import unittest
from datetime import timedelta
from types import SimpleNamespace

from fastapi import HTTPException
from sqlalchemy import create_engine, func, select, update

from app.services.chat import ChatService, chunks, documents, metadata, sessions


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0] + [0.0] * 767 for _ in texts]

    def embed_query(self, text):
        return [1.0] + [0.0] * 767


class FakeLLM:
    def invoke(self, messages):
        return SimpleNamespace(content="Python experience [1].")


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "TEST_DATABASE_URL is required")
class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(os.environ["TEST_DATABASE_URL"])
        cls.chat = ChatService(cls.engine, FakeEmbeddings(), FakeLLM())
        cls.chat.initialize()

    def setUp(self):
        self.tokens = []
        self.chat.embeddings = FakeEmbeddings()
        self.chat.llm = FakeLLM()

    def tearDown(self):
        for token in self.tokens:
            self.chat.end_session(token)

    def session(self):
        state = self.chat.create_session()
        self.tokens.append(state["token"])
        return state

    def expire(self, token):
        with self.engine.begin() as connection:
            connection.execute(update(sessions).where(
                sessions.c.token_hash == self.chat.token_hash(token)
            ).values(expires_at=func.clock_timestamp() - timedelta(seconds=1)))

    @staticmethod
    def pdf():
        import pymupdf
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "Alice has Python and FastAPI experience.")
            return pdf.tobytes()

    def test_fixed_deadline_and_session_isolation(self):
        a, b = self.session(), self.session()
        state = self.chat.upload(a["token"], [("alice.pdf", self.pdf())])
        self.assertEqual(a["expires_at"], state["expires_at"])
        self.assertEqual(self.chat.status(b["token"])["documents"], [])
        ids = [doc["id"] for doc in state["documents"]]
        with self.assertRaises(HTTPException) as error:
            self.chat.query(b["token"], "Who knows Python?", ids)
        self.assertEqual(error.exception.status_code, 404)
        answer = self.chat.query(a["token"], "Who knows Python?", ids)
        self.assertEqual(answer["sources"][0]["filename"], "alice.pdf")

    def test_cleanup_cascades_and_preserves_other_sessions(self):
        a, b = self.session(), self.session()
        state = self.chat.upload(a["token"], [("alice.pdf", self.pdf())])
        ids = [doc["id"] for doc in state["documents"]]
        self.expire(a["token"])
        with self.assertRaises(HTTPException) as error:
            self.chat.status(a["token"])
        self.assertEqual(error.exception.status_code, 410)
        self.chat.cleanup()
        with self.engine.connect() as connection:
            self.assertEqual(connection.scalar(select(func.count()).select_from(chunks).where(chunks.c.document_id.in_(ids))), 0)
            self.assertEqual(connection.scalar(select(func.count()).select_from(documents).where(documents.c.id.in_(ids))), 0)
        self.chat.status(b["token"])

    def test_upload_finishing_after_expiry_does_not_insert(self):
        state = self.session()
        owner = self
        class ExpiringEmbeddings(FakeEmbeddings):
            def embed_documents(self, texts):
                owner.expire(state["token"])
                owner.chat.cleanup()
                return super().embed_documents(texts)
        self.chat.embeddings = ExpiringEmbeddings()
        with self.assertRaises(HTTPException) as error:
            self.chat.upload(state["token"], [("late.pdf", self.pdf())])
        self.assertEqual(error.exception.status_code, 410)

    def test_answer_finishing_after_expiry_is_rejected(self):
        state = self.session()
        uploaded = self.chat.upload(state["token"], [("alice.pdf", self.pdf())])
        owner = self
        class ExpiringLLM:
            def invoke(self, messages):
                owner.expire(state["token"])
                return SimpleNamespace(content="Should not reach the visitor")
        self.chat.llm = ExpiringLLM()
        with self.assertRaises(HTTPException) as error:
            self.chat.query(state["token"], "Who knows Python?", [uploaded["documents"][0]["id"]])
        self.assertEqual(error.exception.status_code, 410)

    def test_expiry_survives_service_restart(self):
        state = self.session()
        restarted = ChatService(self.engine, FakeEmbeddings(), FakeLLM())
        self.assertEqual(restarted.status(state["token"])["expires_at"], state["expires_at"])
        self.expire(state["token"])
        restarted.cleanup()
        with self.assertRaises(HTTPException):
            restarted.status(state["token"])

    def test_http_upload_query_and_expired_session(self):
        from fastapi.testclient import TestClient
        from app.main import app
        app.state.chat = self.chat
        client = TestClient(app)
        response = client.post("/api/v1/chat/sessions")
        self.assertEqual(response.status_code, 201)
        token = response.json()["token"]
        self.tokens.append(token)
        headers = {"Authorization": "Bearer " + token}
        self.assertEqual(client.get("/api/v1/chat/session").status_code, 401)
        upload = client.post("/api/v1/chat/documents", headers=headers,
                             files=[("files", ("alice.pdf", self.pdf(), "application/pdf"))])
        self.assertEqual(upload.status_code, 200, upload.text)
        answer = client.post("/api/v1/chat/messages", headers=headers, json={
            "question": "Who knows Python?", "document_ids": [upload.json()["documents"][0]["id"]],
        })
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.headers["cache-control"], "no-store")
        self.expire(token)
        self.assertEqual(client.get("/api/v1/chat/session", headers=headers).status_code, 410)
        self.assertEqual(client.delete("/api/v1/chat/session", headers=headers).status_code, 204)


if __name__ == "__main__":
    unittest.main()
