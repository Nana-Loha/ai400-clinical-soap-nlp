# 🏥 Clinical SOAP Summarization with ICD-10 RAG

**AI 400 — Natural Language Processing | Bellevue College | March 2026**

A production-aware clinical NLP pipeline that de-identifies patient data, generates structured SOAP summaries using a large language model, and suggests ICD-10 diagnosis codes via Retrieval-Augmented Generation.

---

## 📋 Project Overview

| Component | Technology |
|-----------|-----------|
| PHI De-identification | spaCy NER (`en_core_web_sm`) + Regex |
| SOAP Summarization | Claude Haiku (`claude-haiku-4-5-20251001`) |
| ICD-10 Retrieval | `all-MiniLM-L6-v2` + FAISS vector index |
| Evaluation | ROUGE-1/2/L, BLEU, ICD Precision@5, Recall@5 |
| Dataset | MT Samples (Kaggle) — 30 clinical notes, 3 specialties |

---

## 🚀 Pipeline

```
Raw Clinical Note
      ↓
[Stage 1] PHI De-identification (spaCy NER + Regex)
      ↓
[Stage 2] Baseline SOAP Summarization (Claude Haiku)
      ↓
[Stage 3] ICD-10 RAG (all-MiniLM-L6-v2 + FAISS → Claude Haiku)
      ↓
Evaluation (ROUGE + BLEU + ICD Precision@5 / Recall@5)
```

---

## 📊 Results

| Metric | Baseline | RAG | Improvement |
|--------|----------|-----|-------------|
| ROUGE-1 | 0.4638 | 0.5039 | +0.0401 ✅ |
| ROUGE-2 | 0.1998 | 0.2113 | +0.0115 ✅ |
| ROUGE-L | 0.3001 | 0.3274 | +0.0273 ✅ |
| BLEU | 0.0866 | 0.1038 | +0.0172 ✅ |
| ICD Precision@5 | 0.0000 | 0.0400 | +0.0400 ✅ |
| ICD Recall@5 | 0.0000 | 0.1500 | +0.1500 ✅ |

> RAG outperforms Baseline on **all 6 metrics**. Most significant gain: ICD Recall@5 improves from 0.00 → 0.15.

---

## 📁 Repository Structure

```
ai400-clinical-soap-nlp/
├── AI400_Final_Project_SUBMIT.ipynb   # Main Jupyter Notebook (with outputs)
├── app.py                             # Streamlit demo app
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## 🖥️ Streamlit Demo App

**Live demo:** *(add your Streamlit Cloud URL here)*

Features:
- 🔒 PHI de-identification with before/after view
- 🤖 SOAP summarization (Subjective, Objective, Assessment, Plan)
- 🔍 ICD-10 RAG with top-5 code suggestions
- ⚙️ Toggle RAG on/off, select specialty

### Run locally

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Set your API key:
```bash
# Create .streamlit/secrets.toml
echo 'ANTHROPIC_API_KEY = "sk-ant-..."' > .streamlit/secrets.toml
```

---

## 📓 Notebook

The main project notebook (`AI400_Final_Project_SUBMIT.ipynb`) contains:

- **Section 0:** Model selection & justification
- **Section 0.5:** Literature review (3 papers)
- **Section 2:** Data preparation (MT Samples dataset)
- **Section 2.1:** PHI de-identification
- **Section 3:** Baseline SOAP summarization
- **Section 4:** ICD-10 RAG pipeline
- **Section 5:** Gold reference set (10 human-written notes)
- **Section 6:** Evaluation & results
- **Section 7:** Visualizations
- **Section 8:** Conclusion & business application

---

## 📚 References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*. https://arxiv.org/abs/1908.10084
- Lewis, P., et al. (2020). Retrieval-Augmented Generation. *NeurIPS 2020*. https://arxiv.org/abs/2005.11401
- Lin, C.-Y. (2004). ROUGE. *ACL Workshop on Text Summarization Branches Out*. https://aclanthology.org/W04-1013
- Devlin, J., et al. (2018). BERT. *arXiv:1810.04805*. https://arxiv.org/abs/1810.04805

---

## ⚠️ HIPAA Note

This project uses the MT Samples dataset which is pre-de-identified for research use. The PHI de-identification pipeline (Stage 1) is implemented as a production-ready HIPAA compliance layer — in a real deployment, this step must run before any clinical note reaches an external API.
