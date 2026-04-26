import streamlit as st
from groq import Groq
import PyPDF2
import os
from rag import load_knowledge_base, get_relevant_info
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
.stApp { background: #f8fafc; color: #1e293b; }
#MainMenu, footer, header { visibility: hidden; }
.hero {
    background: linear-gradient(135deg, #1e40af 0%, #0891b2 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(30, 64, 175, 0.3);
}
.hero h1 { color: white; font-size: 2.4rem; font-weight: 700; margin: 0; }
.hero p { color: rgba(255,255,255,0.85); font-size: 1rem; margin: 0.5rem 0 0 0; }
.badge {
    background: rgba(255,255,255,0.2);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    display: inline-block;
    margin: 0.3rem;
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
}
</style>
""", unsafe_allow_html=True)

# Initialize
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

if "knowledge_base" not in st.session_state:
    with st.spinner("Loading medical knowledge base..."):
        st.session_state.knowledge_base = load_knowledge_base()

# Hero
st.markdown("""
<div class="hero">
    <h1>🏥 Health AI Explainer</h1>
    <p>Get a clear, simple explanation of your medical test results</p>
    <br>
    <span class="badge">🌍 3 Languages</span>
    <span class="badge">📁 Any File Format</span>
    <span class="badge">🛡️ Safety Detection</span>
    <span class="badge">🤖 AI Powered</span>
</div>
""", unsafe_allow_html=True)

# Mode selector — hidden in expander
with st.expander("⚙️ Settings"):
    mode = st.radio("Mode", ["👤 Patient Mode", "🔬 Research Mode"], horizontal=True)
    st.caption("Patient Mode: simple results. Research Mode: full AI analysis details.")

# Input
tab1, tab2, tab3 = st.tabs(["📝 Paste Text", "📁 Upload File", "📷 Take Photo"])
health_data = ""

with tab1:
    health_data_text = st.text_area(
        "Paste your results:",
        height=180,
        placeholder="Example:\nHemoglobin: 11.2 g/dL\nWBC: 11,500\nGlucose: 126 mg/dL",
        label_visibility="collapsed"
    )
    if health_data_text:
        health_data = health_data_text

with tab2:
    uploaded_file = st.file_uploader(
        "Upload file",
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
        st.success(f"✅ {uploaded_file.name} uploaded!")

with tab3:
    if not st.session_state.get("camera_active"):
        st.info("📷 Click below to activate camera")
        if st.button("📷 Activate Camera"):
            st.session_state.camera_active = True
            st.rerun()
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Flip Camera"):
                st.rerun()
        with col2:
            if st.button("❌ Turn Off Camera"):
                st.session_state.camera_active = False
                st.rerun()
        camera_photo = st.camera_input("", label_visibility="collapsed")
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
language = st.selectbox("🌍 Language", ["English", "French", "Arabic"])

# Safety check function
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
        # Safety warnings always show
        if not health_data.startswith("IMAGE:"):
            dangerous = check_dangerous_values(health_data)
            if dangerous:
                st.markdown("### 🚨 Urgent — Please See a Doctor")
                for w in dangerous:
                    st.error(w)

        placeholder = st.empty()
        placeholder.info("🤖 Analyzing your results... please wait")

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
            placeholder.empty()
            st.success("✅ Analysis Complete!")
            st.markdown("### 📋 Your Results Explained")
            st.markdown(final_text)
            st.session_state.chat_history = [
                {"role": "user", "content": "Here are my medical results (image)"},
                {"role": "assistant", "content": final_text}
            ]

        else:
            result = run_multi_agent(
                health_data,
                st.session_state.knowledge_base,
                language,
                api_key
            )
            placeholder.empty()

            # PATIENT MODE — clean and simple
            st.success("✅ Analysis Complete!")
            st.markdown("### 📋 Your Results Explained")
            st.markdown(result["report"])

            # RESEARCH MODE — full details
            if "Research Mode" in mode:
                st.divider()
                st.markdown("### 🔬 Research Details")

                evaluation = run_evaluation(
                    health_data,
                    result["report"],
                    language,
                    api_key
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🛡️ Safety Score", f"{evaluation['safety_score']}/10")
                with col2:
                    st.metric("🤖 Agents Used", "2")
                with col3:
                    st.metric("🌍 Language", language)

                with st.expander("🔬 Agent Analysis"):
                    st.markdown("#### Extracted Values")
                    st.markdown(result["extracted"])
                    st.divider()
                    st.markdown("#### Safety Assessment")
                    st.markdown(result["safety"])
                    st.divider()
                    st.markdown("#### Quality Review")
                    st.markdown(result["review"])

                with st.expander("📊 Evaluation Details"):
                    st.markdown(evaluation["llm_scores"])
                    st.divider()
                    for item in evaluation["safety_feedback"]:
                        st.markdown(item)

            st.session_state.chat_history = [
                {"role": "user", "content": f"Here are my medical results:\n{health_data}"},
                {"role": "assistant", "content": result["report"]}
            ]

        st.session_state.show_chat = True
        st.warning("⚠️ For informational purposes only. Always consult a qualified doctor.")

# Chat
if st.session_state.get("show_chat"):
    st.divider()
    st.markdown("### 💬 Ask a Follow-up Question")

    for msg in st.session_state.chat_history[2:]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

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