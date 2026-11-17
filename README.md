# CV Supporter

A temporary resume Q&A workspace built with FastAPI, PostgreSQL/pgvector, Ollama and React/Vite. The original manual and LangChain exercises remain in `app/services`. The public chat uses LangChain for splitting, embeddings and generation, with explicit SQLAlchemy/pgvector persistence to enforce session ownership and cascading deletion.

API and frontend are separate sibling projects:

```text
Documents/
  cv-suporter/        # API, database and Ollama configuration
  cv-suporter-front/  # Independent React app and frontend Dockerfile
```

The local Compose file references `../cv-suporter-front`. Keep the projects side by side when running the complete local stack. They can be versioned and deployed independently.

## Run all four services locally

Install and start Docker Desktop, then run from the repository root:

```sh
docker compose up --build -d
docker compose logs -f ollama
```

Open http://localhost:3000. API docs: http://localhost:8000/docs.

The first startup downloads `nomic-embed-text` and `qwen3:1.7b` into a persistent Ollama volume. The API waits for both models and PostgreSQL to become healthy. Downloads can take several minutes. CPU inference is supported; speed depends on hardware. Docker Desktop needs enough memory for both the models and the other services.

Compose supplies its own internal service URLs; an existing host-oriented `.env` is not used by Compose. Database credentials in Compose are for local development only. PostgreSQL and Ollama are not exposed on host ports. Existing `postgres_data` is retained; its initialized credentials must match the Compose values.

## Session lifecycle

- Opening the page does not start a session. The first upload first calls `POST /api/v1/chat/sessions`, starting the fixed 1,800-second window.
- The random bearer token is an anonymous session capability, not a user login. The database stores only its SHA-256 hash. The browser stores the token in sessionStorage.
- All uploaded documents and chunks belong to a session. Selecting document IDs never bypasses ownership checks.
- Refreshing the page restores the original deadline and uploaded document list. Messages are held in memory and are not restored after reload.
- The timer uses the server's expiry and clock offset. The API enforces the deadline independently of the browser.
- Expired requests receive HTTP 410. Uploads and answers are checked again after model calls, preventing late inference from restoring or returning expired data.
- The frontend clears documents and messages at expiry and requests immediate deletion. A server-side sweep deletes expired session rows every five seconds, cascading to documents, text chunks and embeddings even when the browser is closed.
- If the API is stopped, deletion resumes on startup. This is logical deletion from the active database, not physical erasure from storage snapshots, backups or PostgreSQL WAL.
- Each visitor's expiry deletes only that visitor's rows. It never truncates the whole database.
- Original PDF files are read temporarily, not retained. Questions are independent; previous chat messages are not sent to the model.
- To continue after expiry, upload the resumes again. New session also deletes the current workspace before resetting it.

Limits: 10 text-based PDFs per session, 5 MB per file, 20 MB per upload, 30 pages per PDF, 500 chunks per upload and 4,000 characters per question. Scanned PDFs need OCR, which is not included.

The original study routes are disabled by default. `ENABLE_STUDY_ROUTES=true` is for local exercises only: those endpoints are outside the session lifecycle. Pre-existing records from those exercises are not removed by this feature.

## Vercel + Render

The local Compose deployment and the hosted deployment use the same application code:

```text
Browser -> Vercel (static React app) -> HTTPS Render API
                                         |-> private PostgreSQL + pgvector
                                         |-> private Ollama + model disk
```

### Render

The root `render.yaml` defines three Docker services: a public API, private PostgreSQL, and private Ollama. It uses paid instance plans and persistent disks; review costs and capacity before applying it. Nothing is automatically deployed by this repository change.

1. Create a Blueprint from this repository.
2. Set `CORS_ORIGINS` to the exact Vercel frontend origin, without a trailing slash. Multiple explicit origins can be comma-separated.
3. Generated database credentials and private service hostnames are wired by the Blueprint. All services use the same region.
4. Allow the model downloads to finish. The API health check covers the database; it can become healthy before Ollama finishes downloading, in which case model requests return a retryable 503.
5. Measure inference latency and memory before sharing the demo. The Ollama service is configured for CPU inference and one model at a time. Scale its plan or move Ollama to suitable dedicated hardware if needed.

The database is self-managed in Docker as requested. The operator owns database maintenance and backup policy. An always-running API is necessary for continuous expiry sweeps.

### Vercel

1. Import the independent frontend repository; use its root as **Root Directory**.
2. Choose Vite, build command `npm run build`, output `dist`.
3. Set `VITE_API_URL` to the Render API's public HTTPS origin and redeploy.
4. Set that Vercel origin in Render's `CORS_ORIGINS`.

Vercel builds the static frontend directly. The frontend Dockerfile serves the same build with Nginx for local Compose or other container hosts; it is not needed for the Vercel static deployment.

### Frontend development

```sh
cd ../cv-suporter-front
npm ci
npm run dev
```

Vite proxies `/api` to localhost:8000. To use a remote API, copy `.env.example` to `.env.local` and set `VITE_API_URL`. This is a public build-time URL, never a secret.

## Verification

```sh
cd ../cv-suporter-front
npm ci
npm run build
npx playwright install chromium
npm test
```

Browser tests mock the API and cover uploads, document selection, sources, reload, countdown expiry and mobile layout. They are not proof of real-model inference.

Backend integration tests require a **disposable** PostgreSQL database with pgvector. Install the root requirements, set `DATABASE_URL` and `TEST_DATABASE_URL` to its SQLAlchemy URL, then run:

```sh
python -m unittest discover -s tests -v
```

Tests use fake model clients but real vector storage and cascading foreign keys. They cover cross-session isolation, expiry cleanup, expiry during inference and service restart.

## AWS EC2 deployment

`docker-compose.aws.yaml` is for the existing EC2 host that already runs the Weather platform and Caddy. It starts only the CV Supporter API and its isolated PostgreSQL database. It joins the existing `weather-infra_default` Docker network so Caddy can proxy to the API. It does not run the frontend or Ollama.

On the EC2 host, create a `.env` file from `.env.example`. Set a long random `POSTGRES_PASSWORD`, `CORS_ORIGINS` to the final Vercel origin, and `OLLAMA_HOST` to a reachable Ollama service. Never commit this file.

Start it with:

```sh
docker compose -f docker-compose.aws.yaml up --build -d
```

Add this site to `/home/ec2-user/weather-platform/weather-infra/Caddyfile` after the API reports healthy:

```caddy
cv-supporter-api.duckdns.org {
    encode gzip
    reverse_proxy cv-supporter-api:8080
}
```

Then reload the existing Caddy container. Only ports 80 and 443 should be public; do not publish PostgreSQL or the API port directly.
