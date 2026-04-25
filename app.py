import streamlit as st
from groq import Groq
import PyPDF2
from rag import load_knowledge_base
from multi_agent import run_multi_agent
from evaluator import run_evaluation

st.set_page_config(
    page_title="Health AI Explainer",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #f8fafc;
    color: #1e293b;
}

#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #1e40af 0%, #0891b2 100%);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(30, 64, 175, 0.3);
}

.hero h1 {
    color: white;
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}

.hero p {
    color: rgba(255,255,255,0.85);
    font-size: 1.1rem;
    margin: 0.8rem 0 0 0;
    font-weight: 300;
}

.badge {
    background: rgba(255,255,255,0.2);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    display: inline-block;
    margin: 0.3rem;
    backdrop-filter: blur(10px);
}

.input-card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
    border: 1px solid #e2e8f0;
}

.result-section {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin: 1rem 0;
    border: 1px solid #e2e8f0;
}

.critical-alert {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 1px solid #fca5a5;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.report-box {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border: 1px solid #7dd3fc;
    border-radius: 12px;
    padding: 2rem;
    margin: 1rem 0;
    line-height: 1.8;
    color: #1e293b;
}

.score-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}

.score-item {
    background: white;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.score-number {
    font-size: 2rem;
    font-weight: 700;
    color: #1e40af;
}

.score-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.3rem;
}

.tag {
    background: #eff6ff;
    color: #1e40af;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 0.5rem;
}

.chat-message-user {
    background: #1e40af;
    color: white;
    padding: 0.8rem 1.2rem;
    border-radius: 12px 12px 4px 12px;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
}

.chat-message-ai {
    background: #f1f5f9;
    color: #1e293b;
    padding: 0.8rem 1.2rem;
    border-radius: 12px 12px 12px 4px;
    margin: 0.5rem 0;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.6;
}

.stButton > button {
    background: linear-gradient(135deg, #1e40af, #0891b2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(30, 64, 175, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(30, 64, 175, 0.4) !important;
}

.stTextArea > div > div > textarea {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
}

.stTextArea > div > div > textarea:focus {
    border-color: #1e40af !important;
    box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.1) !important;
}

.stSelectbox > div > div {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
}

.disclaimer {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #92400e;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize
import os
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

if "knowledge_base" not in st.session_state:
    with st.spinner("Loading medical knowledge base..."):
        st.session_state.knowledge_base = load_knowledge_base()

# Hero section
st.markdown("""
<div class="hero">
    <h1>🏥 Health AI Explainer</h1>
    <p>Get a clear, safe, AI-powered explanation of your medical test results</p>
    <br>
    <span class="badge">🔬 RAG Knowledge Base</span>
    <span class="badge">🤖 Multi-Agent AI</span>
    <span class="badge">🛡️ Safety Detection</span>
    <span class="badge">🌍 3 Languages</span>
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-card">', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📝 Paste Text", "📁 Upload File", "📷 Take Photo"])

health_data = ""

with tab1:
    health_data_text = st.text_area(
        "Paste your blood test or medical results:",
        height=180,
        placeholder="Example:\nHemoglobin: 11.2 g/dL\nWBC: 11,500\nGlucose: 126 mg/dL\nBlood Pressure: 120/80 mmHg",
        label_visibility="collapsed"
    )
    if health_data_text:
        health_data = health_data_text

with tab2:
    uploaded_file = st.file_uploader(
        "Upload medical file",
        type=["pdf", "docx", "xlsx", "jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()
        if file_type == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                health_data += page.extract_text()
        elif file_type == "docx":
            from docx import Document
            doc = Document(uploaded_file)
            for paragraph in doc.paragraphs:
                health_data += paragraph.text + "\n"
        elif file_type == "xlsx":
            import pandas as pd
            df = pd.read_excel(uploaded_file)
            health_data = df.to_string()
        elif file_type in ["jpg", "jpeg", "png"]:
            import base64
            from PIL import Image
            import io
            img = Image.open(uploaded_file)
            img = img.resize((800, 600))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            health_data = f"IMAGE:{image_data}"
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")

with tab3:
    if not st.session_state.get("camera_active"):
        st.info("📷 Click below to activate your camera")
        if st.button("📷 Activate Camera"):
            st.session_state.camera_active = True
            st.rerun()
    else:
        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            if st.button("🔄 Flip Camera"):
                st.session_state.camera_facing = "environment" if st.session_state.get("camera_facing") == "user" else "user"
                st.rerun()
        with col_cam2:
            if st.button("❌ Turn Off Camera"):
                st.session_state.camera_active = False
                st.rerun()

        camera_photo = st.camera_input(
            "",
            label_visibility="collapsed"
        )
        if camera_photo:
            import base64
            from PIL import Image
            import io
            img = Image.open(camera_photo)
            img = img.resize((800, 600))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            health_data = f"IMAGE:{image_data}"
            st.success("✅ Photo captured!")

st.markdown("---")
language = st.selectbox(
    "🌍 Explanation language",
    ["English", "French", "Arabic"],
    label_visibility="visible"
)

st.markdown('</div>', unsafe_allow_html=True)

# Safety check
def check_dangerous_values(text):
    import re
    warnings = []
    def extract_number(pattern, text):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
        return None
    glucose = extract_number(r'glucose[:\s]+([\d,]+\.?\d*)', text)
    if glucose:
        if glucose > 200:
            warnings.append(f"🚨 Glucose critically HIGH ({glucose} mg/dL) — Normal: 70-100 mg/dL")
        elif glucose < 50:
            warnings.append(f"🚨 Glucose critically LOW ({glucose} mg/dL) — Normal: 70-100 mg/dL")
    hemoglobin = extract_number(r'hemoglobin[:\s]+([\d,]+\.?\d*)', text)
    if hemoglobin:
        if hemoglobin < 7:
            warnings.append(f"🚨 Hemoglobin critically LOW ({hemoglobin} g/dL) — Normal: 12-17.5 g/dL")
        elif hemoglobin > 20:
            warnings.append(f"🚨 Hemoglobin critically HIGH ({hemoglobin} g/dL) — Normal: 12-17.5 g/dL")
    wbc = extract_number(r'wbc[:\s]+([\d,]+\.?\d*)', text)
    if wbc:
        if wbc > 30000:
            warnings.append(f"🚨 WBC critically HIGH ({wbc}) — Normal: 4,500-11,000")
        elif wbc < 2000:
            warnings.append(f"🚨 WBC critically LOW ({wbc}) — Normal: 4,500-11,000")
    platelets = extract_number(r'platelets[:\s]+([\d,]+\.?\d*)', text)
    if platelets:
        if platelets < 50000:
            warnings.append(f"🚨 Platelets critically LOW ({platelets}) — Normal: 150,000-400,000")
    bp = extract_number(r'blood pressure[:\s]+([\d,]+)', text)
    if bp:
        if bp > 180:
            warnings.append(f"🚨 Blood Pressure critically HIGH ({bp}) — Normal: below 120")
    return warnings

# Analyze button
if st.button("🔍 Explain My Results", use_container_width=True):
    if health_data.strip() == "":
        st.warning("Please enter or upload your health data first.")
    else:
        # Safety warnings
        if not health_data.startswith("IMAGE:"):
            dangerous = check_dangerous_values(health_data)
            if dangerous:
                st.markdown('<div class="critical-alert">', unsafe_allow_html=True)
                st.markdown("### 🚨 Critical Values Detected — Seek Medical Attention")
                for w in dangerous:
                    st.error(w)
                st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("🤖 AI agents analyzing your results..."):
            if health_data.startswith("IMAGE:"):
                image_data = health_data.replace("IMAGE:", "")
                prompt = f"You are a helpful health educator. Explain these medical results clearly in {language}. Explain each value, say if normal/high/low, give wellness tips, remind to see doctor."
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        {"type": "text", "text": prompt}
                    ]}]
                )
                final_text = response.choices[0].message.content
                st.success("✅ Analysis Complete!")
                st.markdown("### 📋 Your Health Report")
                st.write(final_text)
                st.session_state.chat_history = [
                    {"role": "user", "content": "Here are my medical results (image)"},
                    {"role": "assistant", "content": final_text}
                ]

            else:
                api_key = client.api_key
                progress = st.progress(0)
                status = st.empty()

                status.markdown("🔬 Agent 1: Extracting values...")
                progress.progress(20)
                result = run_multi_agent(health_data, st.session_state.knowledge_base, language, api_key)

                status.markdown("📊 Evaluating results...")
                progress.progress(80)
                evaluation = run_evaluation(health_data, result["report"], language, api_key)

                progress.progress(100)
                status.empty()

                # Main report — most prominent
                st.success("✅ Analysis Complete!")
                st.markdown("### 📋 Your Health Report")
                st.write(result["report"])

                # Scores
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🛡️ Safety Score", f"{evaluation['safety_score']}/10")
                with col2:
                    st.metric("🤖 Agents Used", "4")
                with col3:
                    st.metric("🌍 Language", language)

                # Hidden details
                with st.expander("🔬 View AI Analysis Details"):
                    st.markdown("#### 🔬 Agent 1 — Extracted Values")
                    st.markdown(result["extracted"])
                    st.divider()
                    st.markdown("#### 🚨 Agent 2 — Safety Assessment")
                    st.markdown(result["safety"])
                    st.divider()
                    st.markdown("#### ✅ Agent 4 — Quality Review")
                    st.markdown(result["review"])

                with st.expander("📊 View Evaluation Scores"):
                    st.markdown("#### 🤖 AI Judge Scores")
                    st.markdown(evaluation["llm_scores"])
                    st.divider()
                    st.markdown("#### 🛡️ Safety Feedback")
                    for item in evaluation["safety_feedback"]:
                        st.markdown(item)

                st.session_state.chat_history = [
                    {"role": "user", "content": f"Here are my medical results:\n{health_data}"},
                    {"role": "assistant", "content": result["report"]}
                ]

            st.session_state.show_chat = True

        st.markdown('<div class="disclaimer">⚠️ For informational purposes only. Always consult a qualified doctor for medical advice.</div>', unsafe_allow_html=True)

# Chat
if st.session_state.get("show_chat"):
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask a Follow-up Question")

    for msg in st.session_state.chat_history[2:]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-ai">{msg["content"]}</div>', unsafe_allow_html=True)

    follow_up = st.text_input("Type your question here...", key="follow_up_input")
    if st.button("Send 💬"):
        if follow_up.strip():
            st.session_state.chat_history.append({"role": "user", "content": follow_up})
            follow_up_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.chat_history
            )
            reply = follow_up_response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)