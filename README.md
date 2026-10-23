# VeriFace AI 🔍

**VeriFace AI** is a reverse facial recognition and web-discovery engine inspired by platforms like **FaceCheck.ID**, **PimEyes**, and **Lenso.ai**. 

Unlike conventional visual search engines that match clothing, background objects, or scene lighting, VeriFace isolates and extracts human facial biometric geometry. It uses deep metric learning to detect, align, and match faces across the open web and private vector archives using 512-dimensional unit hypersphere embeddings.

---

## ⚡ Key Capabilities

### 1. Dual-Engine Web Facial Discovery (Yandex CBIR + Google Lens)
* **Yandex Images Biometric Index**: Queries Yandex's deep Content-Based Image Retrieval (CBIR) facial database using a **35% padded head crop**, surfacing real facial matches across social networks, public profiles, LinkedIn, TikTok, blogs, and news.
* **Google Lens Dual-Pass**: Concurrently searches the padded head crop and the full image context to capture exact publications, articles, and duplicate reposts across the live internet.
* **Always-On Candidate Discovery**: Discovers 100+ candidates per query, eliminating empty "0 Matches Found" results.

### 2. High-Precision Facial AI Pipeline
* **RetinaFace-R50 Multi-Pass Scanner**: Multi-scale scanning (`640x640`, `1024x1024`, `320x320`, `480x480`), CLAHE shadow/contrast enhancement for backlit photos, and 4-way rotation invariance (90°, 180°, 270°).
* **Closed-Form Umeyama Similarity Transform**: Uses pure NumPy SVD with reflection checking to warp faces to the canonical InsightFace 112x112 template, ensuring perfect convolutional alignment.
* **ArcFace-R100 (512D Embeddings)**: Employs Additive Angular Margin Loss on a 512-dimensional unit hypersphere for maximum inter-class variance and discrimination.
* **Multi-Face Candidate Scanning**: Scans **all faces** in candidate images (e.g. group photos, party pictures, event galleries) so the subject is detected even in crowded shots.

### 3. Calibrated Confidence Tiers
Results are calibrated across an intuitive, transparent 0–100 scale:
* 🟢 **Definite Match** (≥ 85% / Cosine ≥ 0.55): Near-certain identity match across different photos and lighting.
* 🔵 **High Confidence** (72% – 84% / Cosine 0.40 – 0.54): Strong biometric match across poses, aging, or expression.
* 🟣 **Strong Lookalike** (58% – 71% / Cosine 0.28 – 0.39): Distinct structural lookalike sharing eye spacing, nose bridge, and jawline.
* 🟡 **Similar Features / Visual Candidate** (35% – 57%): Closest visual candidates found online with direct links to source pages.

### 4. Modern Web Interface
* **Next.js 15 & Tailwind CSS**: Dark-mode glassmorphic interface with micro-animations.
* **Global Clipboard Paste (`Ctrl+V`)**: Paste any image directly from your clipboard to instantly start scanning.
* **Live Progress Bar**: Displays real-time inference stages (RetinaFace Alignment → ArcFace Embedding → Web Discovery → Neural Re-ranking).
* **Direct Source Links**: Every candidate card displays the host domain, confidence percentage, similarity score, and a direct link to the web page where the photo was discovered.

---

## 🚀 One-Click Launch (Windows PowerShell)

Run the included startup script:

```powershell
.\start-veriface.ps1
```

This launches both servers concurrently:
* **Frontend Web App**: [http://localhost:3000](http://localhost:3000)
* **FastAPI Backend**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🛠️ Manual Installation & Setup

### Prerequisites
* **Python 3.10 – 3.12**
* **Node.js 18+** & npm
* *(Optional)* A free [SerpApi](https://serpapi.com) API key (100 free searches/month, zero credit card required).

---

### Backend Setup

1. **Navigate to the backend directory and activate the virtual environment**:
   ```powershell
   cd backend
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in `backend/`:
   ```env
   SERPAPI_API_KEY=your_serpapi_key_here
   ```
   *(Note: You can also enter and save your SerpApi key directly in the web UI via the Settings ⚙️ modal).*

4. **Start the API Server**:
   ```powershell
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

---

### Frontend Setup

1. **Navigate to the frontend directory**:
   ```powershell
   cd frontend
   ```

2. **Install dependencies**:
   ```powershell
   npm install
   ```

3. **Configure Environment**:
   Create a `.env.local` file in `frontend/`:
   ```env
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
   NEXT_PUBLIC_SERPAPI_KEY=your_serpapi_key_here
   ```

4. **Start the Development Server**:
   ```powershell
   npm run dev
   ```

5. Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 🧪 Testing & Verification

Run the Python test suite to verify the pipeline:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

---

## 🏛️ Architecture & Tech Stack

| Layer | Technology | Description |
|---|---|---|
| **Frontend** | Next.js 15 (React 19), Tailwind CSS, TypeScript | Glassmorphic UI, `Ctrl+V` paste support, live stage tracker |
| **Backend** | FastAPI, Python 3.12, Uvicorn | Async REST API, CORS-enabled, thread-safe model inference |
| **Face Detection** | RetinaFace-R50 (`insightface`) | Multi-scale pyramid, 5-point facial landmark regression |
| **Alignment** | Umeyama Similarity Transform | Closed-form least-squares 2D affine transformation |
| **Feature Extraction** | ArcFace-R100 (`insightface`) | 512-D normalized hypersphere biometric vectors |
| **Vector Storage** | Qdrant & SQLite (WAL mode) | Embedded disk-backed vector storage with cosine metric |
| **Web Retrieval** | Yandex Images CBIR + Google Lens (SerpApi) | Dual-engine visual and biometric candidate discovery |

---

## 📄 License

MIT License. Designed and built for privacy-conscious biometric search, OSINT investigations, and facial geometry research.
