# 🚀 Personalized AI News Recommendation & Chat System (NewsPersona)

[![Docker Compose](https://img.shields.io/badge/Docker_Compose-v3.8+-blue?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PostgreSQL Vector](https://img.shields.io/badge/PostgreSQL_pgvector-v0.2.4-blue?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Inference Engine](https://img.shields.io/badge/Inference-vLLM_/_Ollama-orange?logo=nvidia&logoColor=white)](#-model-serving--fine-tuning)

An end-to-end, ultra-modern, production-grade personalized AI system that curates, recommends, clusters, and summarizes news using local LLM inference engines, vector similarity searches, dynamic user persona modeling, and workflow automation.

---

## 🗺️ System Architecture

This system is built using a highly decoupled microservices architecture containerized through Docker Compose. It leverages PostgreSQL (`pgvector`) for vector search, vLLM / Ollama for high-throughput local LLM inference, FastAPI as a robust backend API, and Streamlit for an interactive user dashboard.

```mermaid
graph TD
    subgraph Client Layer
        User([🧑‍💻 User])
        Streamlit[🎨 Streamlit Dashboard<br/>Port 8501]
        FastAPIPages[🖥️ FastAPI Web Portal<br/>Port 8092]
    end

    subgraph Backend Services
        FastAPI[⚡ FastAPI Core API<br/>Port 8092 / 8000]
        vLLM[🚀 vLLM Inference Engine<br/>Qwen2.5-7B-Instruct-AWQ<br/>Port 8888]
        Ollama[🦙 Ollama Inference Engine<br/>qwen3.5 & nomic-embed<br/>Port 11435]
    end

    subgraph Data & Pipeline Layer
        Postgres[🐘 PostgreSQL + pgvector<br/>Port 5432]
        Redis[🧠 Redis Caching & Session<br/>Port 6379]
        RabbitMQ[🐇 RabbitMQ Message Broker<br/>Port 5672 / 15672]
        n8n[🔄 n8n Workflow Automation<br/>Port 5678]
    end

    subgraph Observability & Utilities
        Grafana[📊 Grafana / Prometheus<br/>Port 3000 / 9090]
        Ngrok[🌐 ngrok Public Tunnel<br/>Port 8088]
    end

    User -->|Interacts| Streamlit
    User -->|Interacts| FastAPIPages
    Streamlit -->|API Requests| FastAPI
    FastAPIPages -->|API Requests| FastAPI
    FastAPI -->|Query Cache / Sessions| Redis
    FastAPI -->|Semantic Vector Search| Postgres
    FastAPI -->|Chat Completions| vLLM
    Streamlit -->|Local Chat / Embeddings| Ollama
    n8n -->|Crawls & Vectorizes Feed| Postgres
    n8n -->|Publishes Events| RabbitMQ
    FastAPI -->|Asynchronous Tasks| RabbitMQ
    Ngrok -->|Secure Public Route| FastAPI
```

---

## ✨ Key Features

### 1. ⚡ FastAPI Production Backend
- **Secure Authentication:** Implements JWT & Bcrypt standard signup, login, and secure sessions.
- **Intelligent Hybrid Conversational RAG pipeline:**
  - **Intent Classification:** Automatically distinguishes between conversational small talk (e.g., "Who are you?", "Thank you") and knowledge-seeking factual requests. Small talk skips DB querying entirely to optimize latency and VRAM.
  - **Confidence Threshold Filtering:** Enforces a similarity threshold of `0.60` for matching documents. Queries below this threshold fall back to general knowledge, preventing LLM hallucinations.
  - **Grounding Rules Integration:** Dynamically attaches strict grounding parameters to the prompt when articles are retrieved, prompting the model to prioritize retrieved facts.
- **Dynamic 3-Tier Persona Engine:** Adapts chatbot system prompts dynamically:
  - *Tier 1:* `USER_CUSTOM_INSTRUCTION` (explicit user override instructions).
  - *Tier 2:* `REFINED_SYSTEM_PROMPT` (automatically generated/adjusted after 5 chat turns to match evolving user habits).
  - *Tier 3:* `INITIAL_PERSONA_PROMPT` (assigned based on chosen interest topics).
- **Proactive Semantic Recommendations:** Triggers a popup every 5 turns, building a semantic search vector from conversation history to recommend the top 3 most relevant articles.
- **Comprehensive Observability:** Logs and returns execution latency for embeddings, DB retrievals, and LLM text generation alongside similarity scores.

### 2. 🎨 Streamlit Interactive Dashboard
- **Topic-Driven Feed Customization:** Let users select 3-4 topics of interest and dynamically calculate preference vectors.
- **Behavioral Personalization:** Blends topic preferences and article-reading history 50/50 using Vector calculations in real time.
- **Dual-Summarization Comparison:** Compares article summaries *with* and *without* the assigned persona instructions side-by-side, showcasing the impact of personalized prompts.
- **Local RAG Chat:** Fully integrated chat window that embeds user inputs, queries nearby vectors via Postgres, and serves answers through local LLM engines.

### 3. 🎯 Model serving & QLoRA Fine-tuning
- **vLLM Engine:** Serves a high-throughput, hardware-optimized quantized model (`Qwen/Qwen2.5-7B-Instruct-AWQ`) using GPU resources.
- **Ollama Engine:** Runs lightweight models (`qwen3.5:4b` and `nomic-embed-text`) locally on edge devices or CPU/GPUs.
- **Finetuning Pipeline (`finetune.py`):**
  - Includes configuration to fine-tune `Qwen/Qwen2.5-7B-Instruct` using **QLoRA (4-bit quantization)** to drastically reduce memory usage (optimized for standard T4 GPUs).
  - Uses `PEFT` (Parameter-Efficient Fine-Tuning) and target modules (`q_proj`, `v_proj`, etc.) to produce compact adapter weights saved separately under `./kietcorn-adapter`.

### 4. 🔄 Workflow Automation & Infrastructure
- **n8n Pipelines:** Integrated as a data ingestion pipeline to automatically scrape RSS channels, vectorize content, and ingest articles directly into Postgres.
- **Message Broker:** RabbitMQ handles event distribution, while Redis holds user session states and temporary caches.

---

## 🛠️ Clone & Project Setup

Follow these steps to set up and run the entire ecosystem on your local machine or server.

### 📋 Prerequisites
- **Operating System:** Linux (Ubuntu 20.04+ recommended)
- **Docker:** Engine 24.0.0+ / Compose v2.20.0+
- **NVIDIA GPU Support (Optional, for vLLM & training):** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed and configured.

### 📥 Step 1: Clone the Repository
```bash
git clone https://github.com/TuanKiet04/FineTune_SML.git
cd FineTune_SML
```

### ⚙️ Step 2: Configure Environment Variables
Create and configure your `.env` file in the root directory. You can copy the values below:
```env
# n8n Authentication
N8N_USER=kietcorn
N8N_PASS=yourpasswordhere

# PostgreSQL Credentials
PG_USER=kietcorn
PG_PASS=yourpasswordhere
PG_DB=kietcorn
DATABASE_URL=postgresql://kietcorn:yourpasswordhere@postgres:5432/kietcorn

# Redis Cache Credentials
REDIS_PASS=yourpasswordhere

# pgAdmin Panel
PGADMIN_EMAIL=admin@example.com
PGADMIN_PASS=yourpasswordhere

# RabbitMQ Message Broker Credentials
RABBITMQ_USER=kietcorn_mq
RABBITMQ_PASS=yourpasswordhere

# JWT Authentication Security Key
JWT_SECRET_KEY=yourkeyhere

# Third-Party API Keys
GEMINI_API_KEY=yourkeyhere
HF_TOKEN=yourkeyhere
NGROK_AUTHTOKEN=yourauthtokenhere
```

### 🚀 Step 3: Run the Stack with Docker Compose

To boot up the complete microservice suite, run:
```bash
docker compose up -d --build
```
> [!NOTE]
> If you do not have an NVIDIA GPU, you may want to comment out the `vllm` service and the `runtime: nvidia` configurations in `docker-compose.yml` to run the stack on a CPU using Ollama.

#### Check Container Status
Verify that all services are online:
```bash
docker compose ps
```

### 📦 Step 4: Run Migrations and Seed Database Personas
The database must be seeded with topic clusters and initial system personas. Run the migrations script inside the FastAPI container:
```bash
docker compose exec fastapi python migrate_and_seed.py
```
*This updates user schemas, configures vector extension plugins, and seeds 10 diverse topic-cluster personas (e.g. Technology, Finance, Sports, Politics) into the PostgreSQL database.*

---

## 📈 Developer Guide & Customization

### Fine-Tuning the LLM (`finetune.py`)
To fine-tune the model on your own dataset:
1. Ensure your training samples are written in standard chat format in `./pattern.jsonl`.
2. Configure settings inside `finetune.py` (e.g., target adapter output, epochs, learning rate).
3. If running locally with virtual environment:
   ```bash
   pip install -r fastapi-app/requirements.txt
   python finetune.py
   ```
4. Compact adapters will be saved to `./kietcorn-adapter`. You can load them onto your base model inside vLLM or Ollama for inference!

### Core Ports & Access Points

| Service | Port | Description |
| :--- | :---: | :--- |
| **FastAPI Backend** | `8092` | REST API, authentication & core logic endpoint (`/docs` for Swagger UI) |
| **Streamlit Dashboard** | `8501` | Front-end graphical interface for user interaction & RAG |
| **n8n Automation** | `5678` | Automations workflow designer |
| **pgAdmin Interface** | `8081` | Database client GUI |
| **RabbitMQ Dashboard** | `15672` | Message broker manager (login with MQ credentials) |
| **Redis Insight** | `5540` | Redis caching inspector |
| **vLLM Engine** | `8888` | High-performance OpenAI-compatible model server |
| **Ollama Local LLM** | `11435` | Local embedding & LLM model manager |

---

## 🛡️ License

This project is open-source and available under the terms of the MIT License.