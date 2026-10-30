"""
Fake Review Detector - web version (Streamlit).
Same trained model and pipeline as the desktop prototype, served in a browser
so it is accessible on any device (phone, tablet, laptop) via a public link.

Files that must sit next to this app.py:
    demo_model.pkl, demo_vectorizer.pkl, demo_scaler.pkl
Optional (enables the Dataset / Results views):
    namibian_reviews_final.csv, results_model_comparison.csv
"""
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="centered")

# IMPORTANT: keep this slang list identical to the one the model was trained with (your 19 words).
SLANG = ['lekker', 'sharp sharp', 'sharp', 'the plug', 'no cap', 'hey', 'proper', 'bru', 'yoh', 'eish', 'giving', 'clean clean', 'boss', 'nxa', 'mooi', 'best best', 'shuu', 'shomafa', 'shiwa nee']

@st.cache_resource
def load_sentiment():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        v = SentimentIntensityAnalyzer()
        return lambda t: v.polarity_scores(t)["compound"]
    except Exception:
        return lambda t: 0.0

sentiment = load_sentiment()

def clean_text(t):
    t = t.lower(); t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s!]", " ", t); return re.sub(r"\s+", " ", t).strip()

def eng_features(text):
    low, cc = text.lower(), len(text)
    return np.array([[len(text.split()), cc, text.count("!"),
        (sum(1 for c in text if c.isupper()) / cc if cc else 0),
        sum(low.count(s) for s in SLANG), sentiment(text)]], dtype=float)

@st.cache_resource
def load_model():
    model = joblib.load("demo_model.pkl")
    vec = joblib.load("demo_vectorizer.pkl")
    scaler = joblib.load("demo_scaler.pkl")
    try:
        names = list(vec.get_feature_names_out())
    except Exception:
        names = list(vec.get_feature_names())
    return model, vec, scaler, names

def classify(model, vec, scaler, text):
    X = vec.transform([clean_text(text)])
    F = scaler.transform(eng_features(text))
    return float(model.predict_proba(hstack([X, csr_matrix(F)]))[:, 1][0])

def top_words(model, vec, names, text, k=8):
    row = vec.transform([clean_text(text)]).tocoo()
    coef = model.coef_[0]
    contribs = [(names[i], v * coef[i]) for i, v in zip(row.col, row.data) if i < len(names)]
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return contribs[:k]

st.title("🔍 Fake Review Detector")
st.caption("Machine learning-based detection for Namibian online consumer reviews")

try:
    model, vec, scaler, FEAT_NAMES = load_model()
    model_ok = True
except Exception as e:
    model_ok = False
    st.error("Could not load the model files. Make sure demo_model.pkl, demo_vectorizer.pkl "
             "and demo_scaler.pkl are in the same folder as app.py.\n\n" + str(e))

tab1, tab2, tab3 = st.tabs(["Detector", "Batch scan", "About"])

with tab1:
    st.subheader("Check a single review")
    examples = {
        "— choose an example —": "",
        "Hype example": "BEST SHOP EVER!!! amazing prices, highly recommend to everyone!!!",
        "Genuine example": "I bought a charger here last week, it works fine and they gave me a receipt.",
    }
    pick = st.selectbox("Try an example (optional)", list(examples.keys()))
    text = st.text_area("Review text", value=examples[pick], height=120,
                        placeholder="Paste a review here...")
    if st.button("Analyse review", type="primary") and model_ok:
        if not text.strip():
            st.warning("Please enter a review first.")
        else:
            prob = classify(model, vec, scaler, text)
            if prob >= 0.5:
                st.error(f"⚠  LIKELY FAKE  —  fake probability {prob:.0%}")
            else:
                st.success(f"✔  LIKELY GENUINE  —  fake probability {prob:.0%}")
            fe = eng_features(text)[0]
            c = st.columns(3)
            c[0].metric("Words", int(fe[0])); c[1].metric("Exclamations", int(fe[2]))
            c[2].metric("Sentiment", f"{fe[5]:.2f}")
            words = top_words(model, vec, FEAT_NAMES, text)
            if words:
                st.markdown("**Words that drove this verdict:**")
                chips = []
                for term, contrib in words:
                    mark = "↑fake" if contrib >= 0 else "↓genuine"
                    chips.append(f"`{term} ({mark})`")
                st.markdown("  ".join(chips))
            else:
                st.info("No known words from this review appeared in the model's vocabulary.")

with tab2:
    st.subheader("Scan many reviews at once")
    up = st.file_uploader("Upload a CSV with a column of review text", type=["csv"])
    if up is not None and model_ok:
        try:
            dfb = pd.read_csv(up)
            textcol = next((c for c in ["text", "review", "Review", "Text", "content"] if c in dfb.columns), None)
            if textcol is None:
                textcol = next((c for c in dfb.columns if dfb[c].dtype == object), None)
            if textcol is None:
                st.error("Could not find a text column in that file.")
            else:
                texts = dfb[textcol].astype(str).tolist()
                probs = [classify(model, vec, scaler, t) for t in texts]
                out = pd.DataFrame({"review": texts,
                                    "fake_probability": np.round(probs, 3),
                                    "verdict": ["LIKELY FAKE" if p >= 0.5 else "LIKELY GENUINE" for p in probs]})
                nfake = int((out["verdict"] == "LIKELY FAKE").sum())
                st.write(f"**{len(out)} reviews scanned** — {nfake} flagged as likely fake, {len(out)-nfake} genuine.")
                st.dataframe(out, use_container_width=True)
                st.download_button("Download results", out.to_csv(index=False).encode(),
                                   "batch_results.csv", "text/csv")
        except Exception as e:
            st.error("Could not process that file: " + str(e))

with tab3:
    st.subheader("About this system")
    st.markdown(
        "This tool uses a supervised machine learning model (Logistic Regression over "
        "TF-IDF and linguistic features) trained on a labelled set of Namibian online "
        "consumer reviews. It estimates whether a review is likely genuine or likely fake "
        "and shows the words behind each decision.\n\n"
        "It is a research prototype and gives a decision-support signal, not a definitive "
        "judgement. Accuracy is lower on short, informal, slang-heavy reviews than on "
        "long formal reviews, which is itself a finding of the study."
    )
    try:
        d = pd.read_csv("results_model_comparison.csv")
        st.markdown("**Model comparison (cross-validated, fake class):**")
        st.dataframe(d, use_container_width=True)
    except Exception:
        pass
