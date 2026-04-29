# 🏥 Health AI Explainer
### A Deployed Multi-Agent Healthcare AI Research Prototype

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-1e40af?style=for-the-badge)](https://health-ai-explainer-rs4yxugmkqmym6vhsbtgi6.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-soukainaderraje-0891b2?style=for-the-badge&logo=github)](https://github.com/soukainaderraje/health-ai-explainer)

---

## 🔬 What Is This?

A deployed Healthcare AI system that explains medical test results (blood tests, lab reports) in clear, simple language — supporting **English, French, and Arabic**.

This is not just an application. It is a **research prototype** designed to investigate:

> *How can LLM-based systems safely and accurately communicate personal health data to non-clinical users — and how do we evaluate whether they are actually doing that well?*

---

## 🚀 Live Demo

👉 **[Try it now](https://health-ai-explainer-rs4yxugmkqmym6vhsbtgi6.streamlit.app)**

Paste any blood test results and get an instant AI-powered explanation in your language.

---

## 🏗️ System Architecture

The system is built around four research components:

### 1. 🔍 RAG Pipeline (Retrieval-Augmented Generation)
- Custom medical knowledge base covering **28 clinical values**
- FAISS vector search with sentence-transformer embeddings
- Semantic retrieval grounds every AI response in verified medical reference data
- Directly addresses the **hallucination problem** in medical AI

### 2. 🤖 Multi-Agent System
Four specialised AI agents working as a pipeline:
| Agent | Role |
|-------|------|
| Agent 1 — Extractor | Extracts and lists all clinical values cleanly |
| Agent 2 — Safety Expert | Checks values against reference ranges |
| Agent 3 — Doctor | Writes warm, patient-friendly report |
| Agent 4 — Reviewer | Reviews report quality and completeness |

### 3. 🛡️ Defence-in-Depth Safety Layer
- Rule-based critical value detection operating **independently of the LLM**
- Instant red-flag alerts for dangerous values before AI explanation begins
- Addresses the known risk of LLMs understating clinical urgency
- Implements **defence-in-depth** — two independent safety mechanisms

### 4. 📊 Automated Evaluation Framework
- **Rule-based safety scoring**: checks if AI correctly communicated danger for each critical value
- **LLM-as-Judge**: second AI evaluates first AI on accuracy, clarity, and completeness
- Produces quantitative scores enabling systematic comparison across languages and approaches

---

## 🌍 Multilingual Support

Full support for **English**, **French**, and **Arabic** across all pipeline stages — including agent system prompts, safety warnings, and evaluation outputs.

---

## 📁 Input Formats Supported

| Format | How |
|--------|-----|
| Plain text | Paste directly |
| PDF | Automatic text extraction |
| Word (.docx) | Paragraph extraction |
| Excel (.xlsx) | Table-to-text conversion |
| Image (JPG/PNG) | Vision AI model reads the image |
| Camera | Take a photo directly in the browser |

---

## 🔧 Technical Stack

```
Frontend:     Streamlit
AI Models:    Groq API (Llama 3.3 70B, Llama 4 Scout Vision)
RAG:          LangChain + FAISS + HuggingFace sentence-transformers
API:          FastAPI + Uvicorn
File Processing: PyPDF2, python-docx, pandas, Pillow
Deployment:   Streamlit Cloud
Version Control: GitHub
```

---

## 📂 Project Structure

```
health-ai-explainer/
├── app.py                  # Main Streamlit application
├── api.py                  # FastAPI REST API
├── rag.py                  # RAG pipeline (knowledge base + search)
├── multi_agent.py          # Four-agent reasoning pipeline
├── agent.py                # Single agent (alternative mode)
├── evaluator.py            # Automated evaluation framework
├── medical_knowledge.txt   # Verified medical reference database
└── requirements.txt        # Dependencies
```

---

## 🔬 Research Questions Addressed

This prototype was built to explore:

1. **Safety**: How can rule-based detection and LLM explanations be combined for defence-in-depth safety in healthcare AI?
2. **Accuracy**: Does RAG grounding in verified medical knowledge reduce hallucination compared to standard prompting?
3. **Evaluation**: Where does LLM-as-Judge succeed and fail as a proxy for human evaluation of AI outputs?
4. **Equity**: Does explanation quality degrade across languages — and what are the implications for multilingual users?

---

## 📊 Key Findings from Preliminary Evaluation

- The LLM consistently underperformed on **glucose-related safety warnings** — describing dangerous values as "elevated" rather than issuing urgent referrals
- Safety scores averaged **5.0–7.5/10** depending on value type and language
- The rule-based safety layer detected **100% of critical values** that the LLM missed or understated
- The LLM-as-Judge evaluation showed **inconsistent reliability** — a finding that motivates further research into automated evaluation methods

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/soukainaderraje/health-ai-explainer.git
cd health-ai-explainer

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
# Create .streamlit/secrets.toml and add:
# GROQ_API_KEY = "your_key_here"

# Run the app
streamlit run app.py

# Run the API (optional)
python -m uvicorn api:app --reload
```

---

## 🌐 API

The system is also exposed as a REST API via FastAPI.

```bash
# Check API status
GET http://localhost:8000/health

# Explain blood test results
POST http://localhost:8000/explain
{
  "health_data": "Glucose: 250 mg/dL\nHemoglobin: 5.5 g/dL",
  "language": "English"
}
```

Full API documentation available at `http://localhost:8000/docs`

---

## 👩‍💻 About

Built by **Soukaina Derraje** — Software Engineer, Data Analyst, and AI Researcher based in Casablanca, Morocco.

This project is part of ongoing research into trustworthy AI evaluation frameworks for high-stakes decision support systems.

📧 soukaina.derraje0@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/soukaina-derraje)

---

> ⚠️ **Disclaimer**: This system is for informational and research purposes only. It does not provide medical advice. Always consult a qualified healthcare professional for medical decisions.
