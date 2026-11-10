# Deployment Guide: Universal Multi-Modal Agentic RAG System

This guide outlines the production deployment architectures, containerization setup, and hosting strategies for the **Universal Multi-Modal Agentic RAG System**.

---

## 1. System Topology & Architecture

The system is decoupled into cloud microservices and local execution runtimes:

```
+-----------------------------------------------------------------------------------+
|                                 USER BROWSER / CLIENT                             |
|                           (Monochrome Research Workspace)                         |
+-----------------------------------------------------------------------------------+
                                          |
                                          | HTTP / REST / SSE (Port 8000)
                                          v
+-----------------------------------------------------------------------------------+
|                            FASTAPI APPLICATION CONTAINER                          |
|                                                                                   |
|  - Ingestion Pipeline: PyMuPDF Page Renderer + IBM Docling Layout Parsing        |
|  - Agentic State Machine: LangGraph Workflow + SQLite Checkpointer                |
|  - Retrieval Engine: Reciprocal Rank Fusion (RRF) + Corrective RAG (CRAG)         |
|  - Storage Mount: /app/data (SQLite DB, Rendered Pages, Chunks, Cropped PNGs)     |
+-----------------------------------------------------------------------------------+
          |                           |                              |
          | HTTP (11434)              | HTTPS (REST API)             | HTTPS (ZeroGPU API)
          v                           v                              v
+--------------------+      +--------------------+      +---------------------------+
|    OLLAMA SERVER   |      |    QDRANT CLOUD    |      |  HF ZEROGPU COLPALI SPACE |
| (nomic-embed-text) |      | (Vector DB Search) |      | (Visual Page Embeddings)  |
+--------------------+      +--------------------+      +---------------------------+
                                      |
                                      | HTTPS (Inference API)
                                      v
                            +--------------------+
                            |      GROQ API      |
                            |  - Qwen 3.8 27B    |
                            |  - GPT-OSS-120B    |
                            +--------------------+
```

---

## 2. Environment Configuration

The application requires the following environment variables. Create a `.env` file in the root directory (refer to [`.env.example`](.env.example)):

| Variable | Required | Description | Example |
| :--- | :---: | :--- | :--- |
| `GROQ_API` | **Yes** | Groq API Key for LLM and VLM inference | `gsk_...` |
| `QDRANT_URL` | **Yes** | Qdrant endpoint (Cloud or self-hosted) | `https://xxxx.cloud.qdrant.io` or `http://localhost:6333` |
| `QDRANT_API_KEY` | **Yes** | Qdrant API key (blank for unauthenticated local instances) | `eyJhbG...` |
| `COLPALI_SPACE` | **Yes** | Remote Hugging Face Space for ColPali visual embeddings | `https://rayyan0-colpali.hf.space` |
| `HF_TOKEN` | **Yes** | Hugging Face Read Token for accessing the ColPali Space | `hf_...` |
| `OLLAMA_HOST` | Optional | Custom Ollama host URL (defaults to `http://localhost:11434`) | `http://ollama:11434` |

---

## 3. Deployment Option 1: Docker Compose (Recommended)

Docker Compose bundles both the **FastAPI Web Server** and the **Ollama Embedding Service** with automatic volume persistence.

### Prerequisites
- Docker Engine `>= 24.0`
- Docker Compose `>= 2.20`

### Steps to Run

1. **Clone the repository and prepare `.env`**:
   ```bash
   git clone <YOUR_REPO_URL>
   cd RAG
   cp .env.example .env
   # Edit .env with your credentials
   nano .env
   ```

2. **Launch all services in detached mode**:
   ```bash
   docker compose up -d --build
   ```

3. **Verify running containers**:
   ```bash
   docker compose ps
   ```

   Expected output:
   ```
   NAME         IMAGE               COMMAND                  SERVICE      STATUS              PORTS
   rag-app      rag-rag-app         "uvicorn backend.ser…"   rag-app      running             0.0.0.0:8000->8000/tcp
   rag-ollama   ollama/ollama:latest "/bin/sh -c ' ollama…"   ollama       running             0.0.0.0:11434->11434/tcp
   ```

4. **Access the application**:
   Navigate to `http://<YOUR_SERVER_IP>:8000` in your browser.

5. **Viewing logs**:
   ```bash
   # View combined application logs
   docker compose logs -f rag-app

   # View Ollama model pull logs
   docker compose logs -f ollama
   ```

6. **Stopping or restarting**:
   ```bash
   docker compose down
   docker compose restart
   ```

---

## 4. Deployment Option 2: Linux Production VM (Ubuntu 22.04 / 24.04 LTS)

For high-performance standalone virtual machines (e.g., AWS EC2, DigitalOcean Droplet, Hetzner Cloud):

### 1. System Packages & Python Setup
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl git libgl1 libglib2.0-0 nginx certbot python3-certbot-nginx

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 2. Install Ollama & Pull Embedding Model
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull nomic-embed-text
ollama pull nomic-embed-text
```

### 3. Clone and Setup Environment
```bash
git clone <YOUR_REPO_URL> /opt/rag
cd /opt/rag

# Create virtual environment and sync dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# Configure environment variables
cp .env.example .env
nano .env
```

### 4. Create Systemd Service
Create `/etc/systemd/system/rag.service`:

```ini
[Unit]
Description=Universal Multi-Modal Agentic RAG Service
After=network.target ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/rag
EnvironmentFile=/opt/rag/.env
Environment="PATH=/opt/rag/.venv/bin:/usr/local/bin:/usr/bin"
Environment="PYTHONPATH=/opt/rag/backend:/opt/rag/backend/src:/opt/rag"
ExecStart=/opt/rag/.venv/bin/uvicorn backend.server:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rag
sudo systemctl start rag
sudo systemctl status rag
```

### 5. Configure Nginx Reverse Proxy with SSL
Create `/etc/nginx/sites-available/rag`:

```nginx
server {
    listen 80;
    server_name rag.yourdomain.com;

    client_max_body_size 100M;  # Support large PDF uploads

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable buffering for progressive typewriter streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

Enable site and configure SSL:
```bash
sudo ln -s /etc/nginx/sites-available/rag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Generate free Let's Encrypt SSL certificate
sudo certbot --nginx -d rag.yourdomain.com
```

---

## 5. Deployment Option 3: Cloud PaaS (Railway / Render)

When deploying to containerized platforms like Railway or Render:

1. **Create Service from GitHub Repo**:
   Select Dockerfile deployment mode.
2. **Mount Persistent Volume**:
   Attach a persistent disk mounted to `/app/data` (Minimum size: 5 GB to 20 GB depending on PDF volume).
   > [!IMPORTANT]
   > The persistent volume at `/app/data` ensures that SQLite conversation checkpoints (`chat_history.db`), raw uploaded PDFs, rendered page images, and cropped figure/table assets survive across redeployments.
3. **Environment Variables**:
   Configure `GROQ_API`, `QDRANT_URL`, `QDRANT_API_KEY`, `COLPALI_SPACE`, `HF_TOKEN`, and `OLLAMA_HOST` in the platform's Environment dashboard.
   *(For Ollama, you can either host Ollama on an external instance or deploy a companion Ollama service on the platform).*

---

## 6. Data Persistence & Backup Strategy

| Directory / File | Description | Backup Priority |
| :--- | :--- | :---: |
| `data/chat_history.db` | SQLite conversation state & LangGraph thread checkpoints | **High** |
| `data/raw/` | Original source PDF documents | **High** |
| `data/rendered_pages/` | Rendered 2x page PNGs | Low (can be re-rendered) |
| `data/figures/` & `data/tables/` | Cropped visual assets | Low (can be re-extracted) |
| `data/chunks/` | Parsed JSONL hierarchy files | Medium |

### Backup Command (Cron Job Example)
```bash
# Nightly backup of database and raw documents
tar -czf /backups/rag_backup_$(date +\%F).tar.gz /opt/rag/data/chat_history.db /opt/rag/data/raw/
```
