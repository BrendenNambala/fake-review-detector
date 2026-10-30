import re
import numpy as np
import joblib
import streamlit as st
from scipy.sparse import hstack, csr_matrix

# ----------------------------------------------------------------------
# These must match the training pipeline EXACTLY, or predictions will be wrong
# ----------------------------------------------------------------------
SLANG = ["lekker", "sharp sharp", "sharp", "the plug", "no cap", "hey",
         "proper", "bru", "yoh", "eish", "giving", "clean clean", "boss"]

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    def sentiment(t):
        return _vader.polarity_scores(t)["compound"]
except Exception:
    def sentiment(t):
        return 0.0


def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s!]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def eng_features(text):
    low, cc = text.lower(), len(text)
    return np.array([[
        len(text.split()),                                  # word_count
        cc,                                                 # char_count
        text.count("!"),                                    # exclamations
        (sum(1 for c in text if c.isupper()) / cc if cc else 0),  # caps_ratio
        sum(low.count(s) for s in SLANG),                   # slang_count
        sentiment(text),                                    # sentiment
    ]], dtype=float)


@st.cache_resource
def load_model():
    model = joblib.load("demo_model.pkl")
    vec = joblib.load("demo_vectorizer.pkl")
    scaler = joblib.load("demo_scaler.pkl")
    return model, vec, scaler


model, vec, scaler = load_model()

# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="centered")

st.title("🔍 Fake Review Detector")
st.caption("Machine learning detection of deceptive reviews in Namibian online consumer reviews")

st.write(
    "Paste a review below. The model estimates whether it is **genuine** or "
    "**suspected fake**, and shows how confident it is."
)

text = st.text_area(
    "Review text",
    height=150,
    placeholder="e.g. BEST SHOP EVER!!! amazing prices!!!",
)

if st.button("Analyse review", type="primary"):
    if not text.strip():
        st.warning("Please paste a review first.")
    else:
        x = vec.transform([clean_text(text)])
        f = scaler.transform(eng_features(text))
        xf = hstack([x, csr_matrix(f)])
        prob = float(model.predict_proba(xf)[0][1])   # probability of FAKE

        if prob >= 0.5:
            st.error(f"### ⚠️  Likely FAKE  —  fake probability {prob:.0%}")
        else:
            st.success(f"### ✅  Likely GENUINE  —  fake probability {prob:.0%}")

        st.progress(prob)

        with st.expander("What the model looked at"):
            feats = eng_features(text)[0]
            st.write({
                "word count": int(feats[0]),
                "character count": int(feats[1]),
                "exclamation marks": int(feats[2]),
                "capitals ratio": round(float(feats[3]), 3),
                "slang words detected": int(feats[4]),
                "sentiment score": round(float(feats[5]), 3),
            })

        st.caption(
            "This is a research baseline (≈74% accuracy, F1 0.53 on the fake class). "
            "It reports a probability, not a guarantee."
        )

st.divider()
st.caption("Honours research prototype · University of Namibia")
