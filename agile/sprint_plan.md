# Agile Sprint Plan — GenAI Customer Intelligence Platform

## Project: GenAI Customer Intelligence Platform
**Methodology**: Scrum (2-week sprints)
**Team Size**: 1 (Swathi K S — Full Stack ML/Data Engineer)
**Total Duration**: 6 Sprints (12 weeks)

---

## Sprint 1 — Project Setup & Data Foundation
**Duration**: Week 1–2
**Goal**: Project scaffolding, data pipeline, EDA

### User Stories
| ID | Story | Points |
|----|-------|--------|
| US-01 | As a developer, I want a structured project with CI/CD so I can deploy reliably | 3 |
| US-02 | As a data engineer, I want a synthetic data generator so I can build without real data | 5 |
| US-03 | As a data analyst, I want to explore the dataset to understand distributions | 3 |

### Tasks
- [x] Initialize Git repo and project structure
- [x] Create virtual environment and requirements.txt
- [x] Build `data_generator.py` (customers, reviews, A/B events)
- [x] Build `preprocessor.py` with feature engineering
- [x] Set up GitHub Actions CI pipeline
- [x] EDA notebook with Pandas Profiling

**Definition of Done**: Data generates successfully, EDA complete, CI pipeline passing.

---

## Sprint 2 — NLP Sentiment Analysis
**Duration**: Week 3–4
**Goal**: Production-grade NLP on customer reviews

### User Stories
| ID | Story | Points |
|----|-------|--------|
| US-04 | As a PM, I want to know overall customer sentiment at a glance | 5 |
| US-05 | As an analyst, I want to see which keywords appear in negative reviews | 3 |
| US-06 | As an ML engineer, I want evaluated NLP metrics (F1, accuracy) | 3 |

### Tasks
- [x] Implement `SentimentAnalyzer` with DistilBERT
- [x] Add rule-based fallback for offline use
- [x] Build `KeywordExtractor` with TF-IDF
- [x] Add evaluation metrics (F1, confusion matrix)
- [x] Unit tests for NLP module

---

## Sprint 3 — Deep Learning Churn Model
**Duration**: Week 5–6
**Goal**: PyTorch churn predictor with MLflow tracking

### User Stories
| ID | Story | Points |
|----|-------|--------|
| US-07 | As a data scientist, I want a DL churn model with AUC > 0.80 | 8 |
| US-08 | As a business stakeholder, I want to understand WHY customers churn (SHAP) | 5 |
| US-09 | As an ML engineer, I want all experiments tracked in MLflow | 3 |

### Tasks
- [x] Design ChurnNet architecture (256→128→64→32→1)
- [x] Implement training loop with validation, early stopping
- [x] Add BatchNorm + Dropout regularization
- [x] SHAP/gradient-based feature importance
- [x] MLflow experiment tracking
- [x] Model save/load functionality

---

## Sprint 4 — A/B Testing & Statistics
**Duration**: Week 7–8
**Goal**: Production-grade experimentation platform

### User Stories
| ID | Story | Points |
|----|-------|--------|
| US-10 | As a product manager, I want to know if our new recommendation engine converts better | 8 |
| US-11 | As a data scientist, I want Bayesian confidence in the result | 5 |
| US-12 | As an analyst, I want a sample size calculator before running experiments | 3 |

### Tasks
- [x] Implement `FrequentistABTest` (z-test, t-test)
- [x] Implement `BayesianABTest` (Beta-Binomial)
- [x] Build `SampleSizeCalculator`
- [x] Add ANOVA and correlation tests
- [x] Unit tests with known-outcome simulations

---

## Sprint 5 — LLM & Gen AI Layer
**Duration**: Week 9–10
**Goal**: LangChain-powered insight generation

### User Stories
| ID | Story | Points |
|----|-------|--------|
| US-13 | As an executive, I want a one-click AI-generated report from data | 8 |
| US-14 | As a business user, I want to ask questions about data in natural language | 5 |
| US-15 | As a developer, I want the system to work without an API key (fallback) | 3 |

### Tasks
- [x] Implement `LLMInsightGenerator` with LangChain
- [x] Prompt engineering for review/AB/churn reports
- [x] Template-based fallback (no API key required)
- [x] RAG-style Q&A over analytics context

---

## Sprint 6 — Cloud, Databricks & Dashboard
**Duration**: Week 11–12
**Goal**: Production deployment and visualization

### User Stories
| ID | Story | Points |
|----|-------|--------|
| US-16 | As a data engineer, I want a Databricks pipeline following medallion architecture | 8 |
| US-17 | As a stakeholder, I want an interactive Streamlit dashboard | 5 |
| US-18 | As a DevOps engineer, I want Azure ML job config and AWS SageMaker deployment | 5 |

### Tasks
- [x] Databricks medallion pipeline (Bronze→Silver→Gold)
- [x] Azure ML job YAML config
- [x] AWS SageMaker inference handler
- [x] Streamlit multi-page dashboard (5 pages)
- [x] End-to-end pipeline runner
- [x] Final documentation and README

---

## Velocity & Retrospective

| Sprint | Planned Points | Completed | Velocity |
|--------|---------------|-----------|----------|
| Sprint 1 | 11 | 11 | 100% |
| Sprint 2 | 11 | 11 | 100% |
| Sprint 3 | 16 | 16 | 100% |
| Sprint 4 | 16 | 16 | 100% |
| Sprint 5 | 16 | 16 | 100% |
| Sprint 6 | 18 | 18 | 100% |
| **Total** | **88** | **88** | **100%** |

---

## SDLC Phases Completed

| Phase | Status | Artifacts |
|-------|--------|-----------|
| 📋 Requirements | ✅ | User stories, acceptance criteria |
| 🎨 Design | ✅ | Architecture diagram, data model |
| 💻 Development | ✅ | All source code |
| 🧪 Testing | ✅ | pytest unit tests, model evaluation |
| 🚀 Deployment | ✅ | Azure ML, AWS SageMaker, Streamlit |
| 📊 Monitoring | 🔄 | MLflow tracking, dashboard |
