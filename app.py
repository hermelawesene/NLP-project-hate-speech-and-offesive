# app.py - MINIMAL WORKING VERSION (copy this entire file)
import os
os.environ["TORCH_LOGS"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import numpy as np
import torch
import re
import nltk
import joblib  # ✅ CRITICAL: Import at top level
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def clean_tweet(tweet):
    tweet = re.sub(r"&[a-z]+;", "", tweet)
    tweet = re.sub(r"@[\w]*", "", tweet)
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
    tokens = tweet.lower().split()
    
    try:
        stopwords = nltk.corpus.stopwords.words("english")
    except:
        nltk.download('stopwords', quiet=True)
        stopwords = nltk.corpus.stopwords.words("english")
    
    stopwords.extend(["#ff", "ff", "rt"])
    
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords]
    return " ".join(tokens)

@st.cache_resource
def load_models():
    # Classical model
    data = joblib.load("classical_model.pkl")
    classical_model = data['model']
    tfidf = data['tfidf']
    
    # Transformer model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert_hate_speech",
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert_hate_speech", local_files_only=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return classical_model, tfidf, model, tokenizer, device

st.set_page_config(page_title="🛡️ Hate Speech Detector", page_icon="🛡️")

st.title("🛡️ Hate Speech Detection System")
st.markdown("""
<div style="background-color:#e3f2fd; padding:15px; border-radius:10px; margin-bottom:20px;">
<h4>⚠️ Ethical Disclaimer</h4>
<p>This tool is for <strong>educational purposes only</strong>. AI systems make mistakes. 
<strong>Never use automated decisions for content removal</strong> without human review.</p>
</div>
""", unsafe_allow_html=True)

try:
    with st.spinner("🚀 Loading models... (first load: 20-40 seconds)"):
        classical_model, tfidf, transformer_model, tokenizer, device = load_models()
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {str(e)}")
    st.info("""
    🔧 HOW TO FIX:
    1. Ensure these files exist in this folder:
       • classical_model.pkl (50-150 MB)
       • distilbert_hate_speech/ folder (with pytorch_model.bin inside)
    2. If missing, re-save models from Colab and download them here
    """)
    st.stop()
except Exception as e:
    st.error(f"❌ Model load failed: {type(e).__name__}: {str(e)[:150]}")
    st.info("💡 Try: pip install --upgrade scikit-learn==1.6.1 joblib")
    st.stop()

def predict_classical(text):
    cleaned = clean_tweet(text)
    vec = tfidf.transform([cleaned])
    pred = classical_model.predict(vec)[0]
    proba = classical_model.predict_proba(vec)[0]
    return int(pred), proba

def predict_transformer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)
    return int(pred), probs

user_input = st.text_area("Enter text to analyze:", placeholder="e.g., 'I hate black people'", height=100)
if st.button("🔍 Analyze", type="primary") and user_input.strip():
    with st.spinner("Analyzing..."):
        c_pred, c_proba = predict_classical(user_input)
        t_pred, t_proba = predict_transformer(user_input)
    
    st.subheader("📝 Input")
    st.info(f'"{user_input}"')
    
    st.subheader("⚖️ Model Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        color = "#ff4444" if c_pred == 0 else "#ffaa00" if c_pred == 1 else "#00cc66"
        label = "⚠️ Hate Speech" if c_pred == 0 else "🔞 Offensive" if c_pred == 1 else "✅ Neither"
        st.markdown(f"""
        <div style="background-color:{color}15; padding:20px; border-radius:10px; border:2px solid {color}">
        <h3 style="margin:0; color:{color}">Classical Model</h3>
        <h2 style="margin:10px 0; color:{color}">{label}</h2>
        <p><strong>Confidence:</strong> {c_proba[c_pred]:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#ff4444" if t_pred == 0 else "#ffaa00" if t_pred == 1 else "#00cc66"
        label = "⚠️ Hate Speech" if t_pred == 0 else "🔞 Offensive" if t_pred == 1 else "✅ Neither"
        st.markdown(f"""
        <div style="background-color:{color}15; padding:20px; border-radius:10px; border:2px solid {color}">
        <h3 style="margin:0; color:{color}">Transformer (DistilBERT)</h3>
        <h2 style="margin:10px 0; color:{color}">{label}</h2>
        <p><strong>Confidence:</strong> {t_proba[t_pred]:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("🛡️ Safety Recommendation")
    final_pred = min(c_pred, t_pred)
    if final_pred == 0:
        st.error("🚨 **FLAG FOR HUMAN REVIEW**\n\nPotential hate speech detected. Requires human moderator review before any action.")
    elif final_pred == 1:
        st.warning("⚠️ **CONTENT WARNING RECOMMENDED**\n\nOffensive language detected. Consider adding warning rather than removal.")
    else:
        st.success("✅ **APPROVE**\n\nNo harmful content detected by either model.")

with st.sidebar:
    st.title("✅ System Status")
    st.success("Models loaded successfully!")
    st.caption(f"Device: {device}")
    st.caption("Console warnings suppressed")