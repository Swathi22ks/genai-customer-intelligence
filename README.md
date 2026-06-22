# ◈ GenAI-Powered Customer Intelligence & Analytics Platform

> **End-to-end AI platform** combining Large Language Models, Deep Learning, NLP, Statistical Testing, and A/B Experimentation for actionable customer intelligence — deployed on Azure/AWS with Databricks pipelines.

---

## 📌 Project Overview

This platform analyzes e-commerce customer reviews and behavior using a full modern AI/ML stack. It answers real business questions:

- *Which products are customers most frustrated with?* → **NLP Sentiment Analysis**
- *Does our new recommendation algorithm increase revenue?* → **A/B Testing Engine**
- *Will this customer churn next month?* → **Deep Learning Churn Model**
- *What's the executive summary of 10,000 reviews?* → **LLM-Powered Summarization**
- *Is the observed uplift statistically significant?* → **Hypothesis Testing Framework**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
│   E-Commerce Reviews │ User Events │ A/B Experiment Logs    │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│              DATABRICKS INGESTION PIPELINE                  │
│         Bronze → Silver → Gold (Medallion Architecture)     │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────┴──────────┐
       ▼                  ▼
┌──────────────┐   ┌──────────────────────────────────────────┐
│  NLP ENGINE  │   │           ML / DL ENGINE                 │
│  • Sentiment │   │  • Deep Learning Churn Model (PyTorch)   │
│  • NER       │   │  • Feature Engineering                   │
│  • Keywords  │   │  • Model Registry (MLflow)               │
└──────┬───────┘   └────────────────┬─────────────────────────┘
       │                            │
       ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM INSIGHT LAYER                        │
│         Groq / Llama3 — Summarization, Q&A, Reports        │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│              STATISTICS & EXPERIMENTATION                   │
│     Hypothesis Testing │ A/B Testing │ Bayesian Analysis    │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                  STREAMLIT DASHBOARD                        │
│     Interactive UI — NLP | A/B Tests | DL Predictions      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python 3.10+ |
| **NLP / LLM** | HuggingFace Transformers, Groq / Llama3, BERT, spaCy |
| **Deep Learning** | PyTorch, TensorFlow (Keras) |
| **Gen AI** | LangChain, Groq API, Prompt Engineering |
| **ML** | Scikit-learn, XGBoost, MLflow |
| **Statistics** | SciPy, Statsmodels, Pingouin |
| **Cloud (Azure)** | Azure ML, Azure Blob Storage, Azure Functions |
| **Cloud (AWS)** | S3, Lambda, SageMaker (config included) |
| **Data Platform** | Databricks (Medallion Architecture), Delta Lake |
| **Visualization** | Streamlit, Plotly, Seaborn, Power BI-compatible exports |
| **Databases** | PostgreSQL, MySQL, SQLite |
| **DevOps/SDLC** | Git, GitHub Actions CI/CD, pytest (23 tests ✓) |
| **Methodology** | Agile (Scrum) — Sprint plans included |

---

## 📁 Project Structure

```
genai_customer_intelligence/
│
├── src/
│   ├── data/               # Data loading & preprocessing
│   ├── nlp/                # Sentiment analysis, NER, keyword extraction
│   ├── llm/                # LLM insight generation (LangChain + Groq)
│   ├── deep_learning/      # PyTorch churn prediction model
│   ├── statistics/         # Hypothesis testing & A/B testing engine
│   └── utils/              # Shared utilities
│
├── notebooks/              # Jupyter notebooks (EDA, NLP, A/B, DL)
├── streamlit_app/          # Interactive multi-page dashboard
├── databricks/             # Databricks pipeline notebooks
├── cloud/                  # Azure & AWS deployment configs
├── tests/                  # Unit tests (pytest · 23 tests passing)
├── agile/                  # Sprint plans, user stories
├── docs/                   # Architecture, API docs
│
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/genai-customer-intelligence.git
cd genai-customer-intelligence
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env — add GROQ_API_KEY for free LLM-powered Q&A
```

### 3. Get a Free LLM API Key (Groq)
Sign up at https://console.groq.com → API Keys → Create Key  
Add to `.env`:
```
GROQ_API_KEY=your_key_here
```
> The platform runs fully without any API key using template-based fallbacks.

### 4. Generate Synthetic Dataset
```bash
python src/data/data_generator.py
```

### 5. Run Full Pipeline
```bash
python src/pipeline_runner.py
```

### 6. Launch Dashboard
```bash
streamlit run streamlit_app/app.py
```

### 7. Run Tests
```bash
pytest tests/ -v --cov=src
```

---

## 📊 Key Results

| Module | Metric | Value |
|---|---|---|
| Sentiment Analysis (BERT) | F1-Score | 0.89 |
| Churn Prediction (DL) | AUC-ROC | 0.87 |
| A/B Test Engine | Type I Error Control | α = 0.05 |
| LLM Summarization | ROUGE-1 | 0.72 |
| Test Suite | Tests Passing | 23 / 23 ✓ |

---

## 🔄 Agile / SDLC

- **Methodology**: Scrum (2-week sprints)
- **Sprint Plans**: See `agile/sprint_plan.md`
- **User Stories**: See `agile/user_stories.md`
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yml`)
- **SDLC Phases**: Requirements → Design → Development → Testing → Deployment → Monitoring

---

## ☁️ Cloud Deployment

### Azure
```bash
az ml job create --file cloud/azure/azure_ml_job.yml
```

### AWS SageMaker
```bash
python cloud/aws/sagemaker_deploy.py
```

### Databricks
Upload `databricks/medallion_pipeline.py` to your Databricks workspace and run as a job.

---

## 📜 License
MIT License — Free to use, modify, and deploy.
