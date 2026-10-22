# VeriFace

VeriFace is a reverse facial-recognition search application. The project is being built in gated steps so that its detection, alignment, and embedding pipeline can be reviewed before database and UI integrations are added.

## Step 1: local services and application skeleton

Copy the example environment settings, then start the data services:

```powershell
Copy-Item .env.example .env
docker compose up -d
```

The services expose PostgreSQL at `localhost:5432`, Qdrant's REST API at `localhost:6333`, and Qdrant gRPC at `localhost:6334`. Both persist data in named Docker volumes.

To run the API locally:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

To run the frontend locally:

```powershell
cd frontend
npm install
npm run dev
```

The API liveness check is available at `http://localhost:8000/health` and the Next.js app at `http://localhost:3000`.

## Project structure

- `backend/`: FastAPI service and pinned runtime requirements.
- `frontend/`: Next.js + TypeScript + TailwindCSS application.
- `docker-compose.yml`: Qdrant and PostgreSQL development infrastructure.

The Python packages for InsightFace, ONNX Runtime, OpenCV, Qdrant, and PostgreSQL are declared. The database schema and search APIs are intentionally deferred to later approved steps.

## Step 2: strict face-inference pipeline

`backend/app/services/face_pipeline.py` now enforces this exact sequence:

1. Decode the upload and reject it if its Laplacian-variance score is below `VERIFACE_BLUR_THRESHOLD` (default `100`).
2. Detect faces with the configured RetinaFace ONNX model and reject a selected face smaller than `112 × 112` pixels.
3. Use RetinaFace's five landmarks to estimate an affine transform to the canonical ArcFace `112 × 112` landmark template.
4. Run the aligned crop through the configured ArcFace ONNX model, require exactly 512 values, and L2-normalize the resulting vector.

Before running the service, obtain approved InsightFace-compatible RetinaFace and ArcFace ONNX files and place them at the configured paths under `backend/models/` (the default names are in `.env.example`). Model files are deliberately gitignored. The default ONNX provider is CPU; configure `VERIFACE_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider` on a CUDA-enabled deployment.

The pipeline is not exposed as a public route yet. That intentional boundary prevents an unauthenticated API from accepting face uploads until the search API and its access controls are added in Step 4.

## Step 3: database and vector-index wiring

The PostgreSQL schema in `backend/app/db/models.py` keeps source-image and page-context URLs in `indexed_faces`, linked by UUID to both `face_profiles` and Qdrant points. `FaceIndexer` creates the Qdrant `face_embeddings` collection as a 512D cosine collection with explicit HNSW settings (`m=16`, `ef_construct=200`), normalizes every vector, and compensates for failed cross-database writes.

After starting PostgreSQL and Qdrant, create the schema and index a vetted local image:

```powershell
cd backend
python -m scripts.index_face `
  --image-file C:\path\to\face.jpg `
  --source-image-url https://example.org/face.jpg `
  --page-context-url https://example.org/page
```

Pass `--profile-id <UUID>` only to add a known, verified sample to an existing profile. The future feedback workflow will be the normal mechanism for enriching a profile.

## Steps 4–5: search, feedback, and UI

`POST /search` accepts multipart field `image` and optional `top_k` (1–100). It runs the strict face pipeline, writes a 24-hour feedback session, searches Qdrant with cosine similarity, and returns only results at or above 60% similarity. `POST /feedback` accepts a `search_id`, `indexed_face_id`, and `is_match`.

A positive feedback event queues background enrichment: the query embedding is stored as a private confirmed sample and updates the profile's normalized running-average master vector. Master/private points resolve to a public representative source before rendering, so every result keeps a direct source-image URL and page-context URL.

The Next.js UI now provides drag-and-drop upload, Detecting → Aligning → Embedding → Searching progress, a responsive masonry results grid, color-coded similarity badges, direct source links, and verification controls.

To test the complete stack, first place approved ONNX files at `backend/models/retinaface_r50_v1.onnx` and `backend/models/arcface_r100_v1.onnx`, then run:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Index at least one vetted image with `python -m scripts.index_face ...` before searching. The UI is at `http://localhost:3000`; API documentation is at `http://localhost:8000/docs`.
