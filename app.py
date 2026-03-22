"""
AI400 Final Project — Clinical SOAP Summarizer with ICD-10 RAG
Deploy on Streamlit Cloud:
  1. Push this file + requirements.txt to GitHub
  2. Go to share.streamlit.io → connect repo
  3. Add ANTHROPIC_API_KEY in Streamlit Cloud Secrets
"""

import re
import json
import time
import requests
import streamlit as st
import anthropic

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical SOAP Summarizer",
    page_icon="🏥",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #028090; }
    .sub-title  { font-size: 1rem; color: #64748b; margin-bottom: 1.5rem; }
    .soap-box   { background: #f0f9ff; border-left: 4px solid #028090;
                  padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
                  margin-bottom: 0.6rem; }
    .soap-label { font-weight: 700; color: #028090; font-size: 0.85rem;
                  text-transform: uppercase; letter-spacing: 0.05em; }
    .icd-badge  { display: inline-block; background: #e0f7f4; color: #028090;
                  border: 1px solid #02c39a; border-radius: 6px;
                  padding: 0.25rem 0.6rem; margin: 0.2rem;
                  font-size: 0.85rem; font-weight: 600; }
    .phi-box    { background: #fff7ed; border-left: 4px solid #f59e0b;
                  padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; }
    .urgent-box { background: #fef2f2; border-left: 4px solid #ef4444;
                  padding: 0.5rem 1rem; border-radius: 0 8px 8px 0; }
    .step-badge { background: #028090; color: white; border-radius: 50%;
                  width: 24px; height: 24px; display: inline-flex;
                  align-items: center; justify-content: center;
                  font-size: 0.8rem; font-weight: 700; margin-right: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL      = "claude-haiku-4-5-20251001"
ICD10_URL  = "https://raw.githubusercontent.com/k4m4/icd10-cm/master/icd10cm_codes_2025.txt"

SYNTHETIC_EXAMPLE = """PATIENT: John Smith
DOB: 03/15/1972    MRN: 456789    Phone: 206-555-0198
Email: jsmith@email.com
Referred by Dr. Emily Johnson at Swedish Medical Center, Seattle.

Chief Complaint: Patient presents with chest pain and shortness of breath for 3 days.
History of hypertension managed with lisinopril 10mg daily. No known drug allergies.
BP 148/92, HR 88 bpm, RR 18, SpO2 97% on room air.
EKG shows normal sinus rhythm. Troponin pending.
Assessment: Possible unstable angina versus GERD.
Plan: Admit for cardiac monitoring, serial troponins, cardiology consult."""

BASELINE_PROMPT = """You are an expert clinical documentation assistant.
Summarize the note accurately and safely.

CRITICAL RULES:
1) Do not fabricate details not present in the note.
2) If information is missing or ambiguous, add it to unclear_items.
3) Output must be valid JSON only — no markdown fences.

Return this JSON schema exactly:
{
  "soap": {
    "subjective": "...",
    "objective": "...",
    "assessment": "...",
    "plan": "..."
  },
  "extracted_entities": {
    "diagnoses": [],
    "medications": [],
    "allergies": []
  },
  "urgent_flags": [],
  "unclear_items": [],
  "confidence_score": 0.0,
  "summary_narrative": "2-3 sentence clinical summary"
}"""

RAG_PROMPT = """You are an expert clinical documentation assistant.
Summarize the note accurately and safely.

CRITICAL RULES:
1) Do not fabricate details not present in the note.
2) icd10_suggestions MUST contain ONLY codes from the RETRIEVED CODES LIST provided.
   If none fit, return an empty list [].
3) Output must be valid JSON only — no markdown fences.

Return exactly this JSON schema:
{
  "soap": {
    "subjective": "...",
    "objective": "...",
    "assessment": "...",
    "plan": "..."
  },
  "extracted_entities": {
    "diagnoses": [],
    "medications": [],
    "allergies": []
  },
  "icd10_suggestions": [
    {"code": "...", "description": "...", "evidence": "brief quote from note"}
  ],
  "urgent_flags": [],
  "unclear_items": [],
  "confidence_score": 0.0,
  "summary_narrative": "2-3 sentence clinical summary"
}"""

# ── PHI De-identification ──────────────────────────────────────────────────────
def deidentify_note(text: str) -> tuple[str, list]:
    """Hybrid spaCy NER + Regex PHI de-identification."""
    deidentified = text
    entities_found = []

    # Try spaCy NER
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        replacements = []
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "DATE", "GPE", "ORG"}:
                replacements.append((ent.start_char, ent.end_char,
                                     f"[{ent.label_}]", ent.text))
        for start, end, placeholder, orig in sorted(replacements, reverse=True):
            deidentified = deidentified[:start] + placeholder + deidentified[end:]
            entities_found.append({"type": "NER", "original": orig,
                                    "replaced_with": placeholder})
    except Exception:
        pass  # spaCy not available — regex only

    # Regex patterns
    patterns = [
        (r"\b\d{3}-\d{2}-\d{4}\b",                              "[SSN]",    "SSN"),
        (r"\bMRN[:\s#]*\d{4,10}\b",                             "[MRN_ID]", "MRN"),
        (r"\b(\d{3}[\-.\s]?\d{3}[\-.\s]?\d{4})\b",             "[PHONE]",  "PHONE"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b","[EMAIL]",  "EMAIL"),
        (r"\bDOB[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",        "[DOB]",    "DOB"),
        (r"\b\d{1,2}/\d{1,2}/\d{4}\b",                          "[DATE]",   "DATE"),
    ]
    for pattern, placeholder, phi_type in patterns:
        matches = re.findall(pattern, deidentified)
        if matches:
            flat = [m if isinstance(m, str) else m[0] for m in matches]
            for m in flat:
                entities_found.append({"type": phi_type, "original": m,
                                        "replaced_with": placeholder})
            deidentified = re.sub(pattern, placeholder, deidentified)

    return deidentified, entities_found

# ── ICD-10 helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_icd10_codes() -> tuple[list, list]:
    """Download and cache ICD-10 codes."""
    try:
        r = requests.get(ICD10_URL, timeout=30)
        r.raise_for_status()
        codes, descs = [], []
        for line in r.text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                codes.append(parts[0])
                descs.append(parts[1])
        return codes, descs
    except Exception:
        return [], []

@st.cache_resource(show_spinner=False)
def build_search_index(codes: tuple, descs: tuple):
    """Build FAISS or NumPy index for ICD-10 retrieval."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(
        list(descs), batch_size=256, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    )
    try:
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype("float32"))
        return {"type": "faiss", "index": index, "embeddings": embeddings}, model
    except Exception:
        return {"type": "numpy", "index": None, "embeddings": embeddings}, model

def retrieve_icd10(query: str, index, codes, descs, model, top_k=5) -> list:
    """Retrieve top-k ICD-10 codes for a query."""
    import numpy as np
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    if index["type"] == "faiss":
        _, I = index["index"].search(q_vec, top_k)
        idxs = I[0].tolist()
    else:
        sims = (index["embeddings"] @ q_vec[0]).astype("float32")
        idxs = np.argsort(-sims)[:top_k].tolist()
    return [{"code": codes[i], "description": descs[i]} for i in idxs]

# ── LLM helpers ────────────────────────────────────────────────────────────────
def parse_json(text: str) -> dict:
    clean = re.sub(r"```json\s*|\s*```", "", text).strip()
    try:
        return json.loads(clean)
    except Exception:
        return {"parse_error": True, "raw": text}

def call_baseline(note: str, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=MODEL, max_tokens=2000, temperature=0,
        system=BASELINE_PROMPT,
        messages=[{"role": "user", "content":
            f"CLINICAL NOTE:\n{note}\n\nReturn ONLY valid JSON."}]
    )
    return parse_json(msg.content[0].text)

def call_rag(note: str, retrieved: list, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    code_block = "\n".join(f"  {r['code']} — {r['description']}" for r in retrieved)
    msg = client.messages.create(
        model=MODEL, max_tokens=2000, temperature=0,
        system=RAG_PROMPT,
        messages=[{"role": "user", "content":
            f"RETRIEVED ICD-10 CODES (use ONLY these):\n{code_block}\n\n"
            f"CLINICAL NOTE:\n{note}\n\nReturn ONLY valid JSON."}]
    )
    return parse_json(msg.content[0].text)

# ── UI Helpers ─────────────────────────────────────────────────────────────────
def render_soap(soap: dict):
    labels = {"subjective": "S — Subjective", "objective": "O — Objective",
              "assessment": "A — Assessment", "plan": "P — Plan"}
    for key, label in labels.items():
        val = soap.get(key, "")
        if val:
            st.markdown(f"""
            <div class="soap-box">
                <div class="soap-label">{label}</div>
                <div>{val}</div>
            </div>""", unsafe_allow_html=True)

def render_entities(entities: dict):
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**🔬 Diagnoses**")
        for d in entities.get("diagnoses", []):
            st.markdown(f"- {d}")
    with cols[1]:
        st.markdown("**💊 Medications**")
        for m in entities.get("medications", []):
            st.markdown(f"- {m}")
    with cols[2]:
        st.markdown("**⚠️ Allergies**")
        for a in entities.get("allergies", []):
            st.markdown(f"- {a}")

# ── Main App ───────────────────────────────────────────────────────────────────
def main():
    # Get API key from Streamlit secrets
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        api_key = None

    # Header
    st.markdown('<div class="main-title">🏥 Clinical SOAP Summarizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI 400 Final Project — PHI De-identification + LLM Summarization + ICD-10 RAG</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Pipeline:**
        1. 🔒 PHI De-identification (spaCy NER + Regex)
        2. 🤖 SOAP Summarization (Claude Haiku)
        3. 🔍 ICD-10 RAG (all-MiniLM-L6-v2 + FAISS)

        **Models:**
        - `claude-haiku-4-5-20251001`
        - `all-MiniLM-L6-v2`

        **Dataset:** MT Samples (Kaggle)
        """)
        st.divider()
        st.markdown("### ⚙️ Settings")
        use_rag = st.toggle("Enable ICD-10 RAG", value=True)
        show_phi = st.toggle("Show PHI de-identification", value=True)
        specialty = st.selectbox("Specialty", [
            "Surgery", "Consult - History and Phy.",
            "Cardiovascular / Pulmonary", "Orthopedic", "Other"
        ])

    # Input
    st.markdown("### 📝 Clinical Note Input")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📋 Load Example", use_container_width=True):
            st.session_state["note_input"] = SYNTHETIC_EXAMPLE

    note_input = st.text_area(
        "Paste clinical note here:",
        value=st.session_state.get("note_input", ""),
        height=200,
        placeholder="Paste a clinical note here, or click 'Load Example' →",
        key="note_input"
    )

    # Process button
    if st.button("🚀 Analyze Note", type="primary", use_container_width=True):
        if not note_input.strip():
            st.warning("Please enter a clinical note.")
            return
        if not api_key:
            st.error("ANTHROPIC_API_KEY not found in Streamlit secrets.")
            return

        # ── Stage 1: PHI De-identification ────────────────────────────────────
        with st.spinner("🔒 Stage 1: De-identifying PHI..."):
            deidentified, phi_entities = deidentify_note(note_input)

        if show_phi:
            st.markdown("### 🔒 Stage 1: PHI De-identification")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original (with PHI)**")
                st.code(note_input[:500] + ("..." if len(note_input) > 500 else ""), language=None)
            with col_b:
                st.markdown("**De-identified (HIPAA-safe)**")
                st.code(deidentified[:500] + ("..." if len(deidentified) > 500 else ""), language=None)

            if phi_entities:
                st.markdown(f'<div class="phi-box">✅ <b>{len(phi_entities)} PHI entities removed</b>: ' +
                    ", ".join(f'<code>{e["type"]}</code>' for e in phi_entities[:8]) +
                    ("..." if len(phi_entities) > 8 else "") + "</div>", unsafe_allow_html=True)
            else:
                st.info("No PHI detected — note appears already de-identified.")

        st.divider()

        # ── Stage 2: Baseline SOAP ─────────────────────────────────────────────
        st.markdown("### 🤖 Stage 2: SOAP Summarization (Claude Haiku)")
        with st.spinner("Generating SOAP summary..."):
            t0 = time.time()
            baseline = call_baseline(deidentified, api_key)
            elapsed = round(time.time() - t0, 1)

        if baseline.get("parse_error"):
            st.error("Failed to parse baseline response.")
        else:
            conf = baseline.get("confidence_score", 0)
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence Score", f"{conf:.0%}")
            col2.metric("Processing Time", f"{elapsed}s")
            col3.metric("Model", "Claude Haiku")

            render_soap(baseline.get("soap", {}))

            with st.expander("🔬 Extracted Entities"):
                render_entities(baseline.get("extracted_entities", {}))

            if baseline.get("urgent_flags"):
                flags = " · ".join(baseline["urgent_flags"])
                st.markdown(f'<div class="urgent-box">🚨 <b>Urgent flags:</b> {flags}</div>',
                            unsafe_allow_html=True)

            narrative = baseline.get("summary_narrative", "")
            if narrative:
                st.info(f"📋 **Summary:** {narrative}")

        # ── Stage 3: ICD-10 RAG ────────────────────────────────────────────────
        if use_rag:
            st.divider()
            st.markdown("### 🔍 Stage 3: ICD-10 RAG")

            with st.spinner("Loading ICD-10 index (first run ~2 min)..."):
                codes, descs = load_icd10_codes()
                if not codes:
                    st.error("Failed to load ICD-10 codes.")
                    return
                index, embed_model = build_search_index(tuple(codes), tuple(descs))

            # Build query from baseline diagnoses
            diags = baseline.get("extracted_entities", {}).get("diagnoses", [])
            query = ", ".join(diags[:5]) if diags else deidentified[:300]
            retrieved = retrieve_icd10(query, index, codes, descs, embed_model)

            with st.expander("🔎 Retrieved ICD-10 Shortlist (top-5)", expanded=False):
                for r in retrieved:
                    st.markdown(f'<span class="icd-badge">{r["code"]}</span> {r["description"]}',
                                unsafe_allow_html=True)

            with st.spinner("Generating RAG-constrained summary..."):
                t0 = time.time()
                rag_result = call_rag(deidentified, retrieved, api_key)
                elapsed_rag = round(time.time() - t0, 1)

            if rag_result.get("parse_error"):
                st.error("Failed to parse RAG response.")
            else:
                st.markdown("#### ICD-10 Suggestions")
                icd_suggestions = rag_result.get("icd10_suggestions", [])
                if icd_suggestions:
                    for s in icd_suggestions:
                        st.markdown(
                            f'<span class="icd-badge">{s.get("code","?")}</span> '
                            f'**{s.get("description","?")}** — *{s.get("evidence","")}*',
                            unsafe_allow_html=True)
                else:
                    st.info("No ICD-10 codes matched the retrieved shortlist for this note.")

                st.markdown("#### RAG-Enhanced SOAP")
                render_soap(rag_result.get("soap", {}))

                with st.expander("🔬 RAG Extracted Entities"):
                    render_entities(rag_result.get("extracted_entities", {}))

                col1, col2 = st.columns(2)
                col1.metric("RAG Processing Time", f"{elapsed_rag}s")
                col2.metric("ICD Codes Suggested", len(icd_suggestions))

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:0.8rem;">
    AI 400 NLP Final Project · Bellevue College · March 2026<br>
    Reimers & Gurevych (2019) · Lewis et al. (2020) · Lin (2004) · Devlin et al. (2018)
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
