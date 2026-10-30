"""
Fake Review Detector - web version (Streamlit), modern styling.
"""
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #f6f8fc; }
#MainMenu, footer {visibility: hidden;}
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
  border-radius: 18px; padding: 26px 28px; color: #fff; margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(30,58,138,.25);
}
.hero h1 { font-size: 26px; font-weight: 700; margin: 0 0 4px; color:#fff; }
.hero p  { font-size: 14px; margin: 0; color: #c7d2fe; }
.verdict { border-radius: 14px; padding: 18px 20px; margin: 6px 0 4px;
           font-weight: 700; font-size: 18px; display:flex; align-items:center;
           justify-content:space-between; }
.verdict small { font-weight:600; font-size:13px; opacity:.85; }
.v-fake    { background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; }
.v-genuine { background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.chip  { font-size:12px; font-weight:600; padding:3px 9px; border-radius:20px; }
.c-fake    { background:#fee2e2; color:#b91c1c; }
.c-genuine { background:#dcfce7; color:#15803d; }
.stat { background:#fff; border:1px solid #e5e9f2; border-radius:14px;
        padding:14px 16px; text-align:center; }
.stat .n { font-size:24px; font-weight:700; color:#1e3a8a; }
.stat .l { font-size:12px; color:#64748b; margin-top:2px; }
.stButton button { border-radius:10px; font-weight:600; }
thead tr th { background:#eef2ff !important; color:#1e293b !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🔍 Fake Review Detector</h1>
  <p>Machine learning-based detection for Namibian online consumer reviews</p>
</div>
""", unsafe_allow_html=True)

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

try:
    model, vec, scaler, FEAT_NAMES = load_model()
    model_ok = True
except Exception as e:
    model_ok = False
    st.error("Could not load the model files. Make sure demo_model.pkl, demo_vectorizer.pkl "
             "and demo_scaler.pkl sit next to app.py.\n\n" + str(e))

tab1, tab2, tab3 = st.tabs(["🔎  Detector", "📂  Batch scan", "ℹ️  About"])

with tab1:
    st.markdown("#### Check a single review")
    examples = {
        "— choose an example —": "",
        "Hype example": "BEST SHOP EVER!!! amazing prices, highly recommend to everyone!!!",
        "Genuine example": "I bought a charger here last week, it works fine and they gave me a receipt.",
    }
    pick = st.selectbox("Try an example (optional)", list(examples.keys()))
    text = st.text_area("Review text", value=examples[pick], height=120,
                        placeholder="Paste a review here...")
    if st.button("Analyse review", type="primary", use_container_width=True) and model_ok:
        if not text.strip():
            st.warning("Please enter a review first.")
        else:
            prob = classify(model, vec, scaler, text)
            if prob >= 0.5:
                st.markdown(f'<div class="verdict v-fake">⚠ Likely fake'
                            f'<small>fake probability {prob:.0%}</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict v-genuine">✔ Likely genuine'
                            f'<small>fake probability {prob:.0%}</small></div>', unsafe_allow_html=True)
            fe = eng_features(text)[0]
            c = st.columns(3)
            for col, n, l in [(c[0], int(fe[0]), "Words"), (c[1], int(fe[2]), "Exclamations"),
                              (c[2], f"{fe[5]:.2f}", "Sentiment")]:
                col.markdown(f'<div class="stat"><div class="n">{n}</div><div class="l">{l}</div></div>',
                             unsafe_allow_html=True)
            words = top_words(model, vec, FEAT_NAMES, text)
            if words:
                st.markdown("###### Words that drove this verdict")
                chips = "".join(
                    f'<span class="chip {"c-fake" if contrib>=0 else "c-genuine"}">'
                    f'{term} {"↑fake" if contrib>=0 else "↓genuine"}</span>'
                    for term, contrib in words)
                st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
            else:
                st.info("No known words from this review appeared in the model's vocabulary.")

with tab2:
    st.markdown("#### Scan many reviews at once")
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
                a, b = st.columns(2)
                a.markdown(f'<div class="stat"><div class="n">{nfake}</div><div class="l">Flagged as fake</div></div>', unsafe_allow_html=True)
                b.markdown(f'<div class="stat"><div class="n">{len(out)-nfake}</div><div class="l">Genuine</div></div>', unsafe_allow_html=True)
                st.markdown("")
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("⬇  Download results", out.to_csv(index=False).encode(),
                                   "batch_results.csv", "text/csv", use_container_width=True)
        except Exception as e:
            st.error("Could not process that file: " + str(e))

with tab3:
    st.markdown("#### About this system")
    st.markdown(
        "This tool uses a supervised machine-learning model — **Logistic Regression** over "
        "**TF-IDF** text features combined with **engineered linguistic features** (sentiment, "
        "length, exclamations and Namibian slang) — trained on a labelled set of Namibian online "
        "consumer reviews. It estimates whether a review is likely genuine or fake and shows the "
        "words behind each decision. It is a research prototype that gives a decision-support "
        "signal, not a definitive judgement."
    )
    st.markdown("###### Best model at a glance")
    acc, f1, rec, auc = "0.809", "0.727", "0.829", "0.835"
    try:
        d = pd.read_csv("results_model_comparison.csv")
        best = d[(d["model"] == "Logistic Regression") & (d["feature_set"] == "TF-IDF + features")].iloc[0]
        acc, f1, rec = f"{best['accuracy']:.3f}", f"{best['f1']:.3f}", f"{best['recall']:.3f}"
    except Exception:
        d = None
    k = st.columns(4)
    for col, n, l in [(k[0], acc, "Accuracy"), (k[1], f1, "F1 (fake)"),
                      (k[2], rec, "Recall (fake)"), (k[3], auc, "ROC AUC")]:
        col.markdown(f'<div class="stat"><div class="n">{n}</div><div class="l">{l}</div></div>',
                     unsafe_allow_html=True)
    st.markdown("")
    if d is not None:
        st.markdown("###### Model comparison")
        st.caption("Five-fold cross-validated scores on the fake class. The highlighted row is the deployed model.")
        disp = d.rename(columns={"feature_set": "Feature set", "model": "Model",
                                 "accuracy": "Accuracy", "precision": "Precision",
                                 "recall": "Recall", "f1": "F1"})
        def hl(row):
            best = (row["Model"] == "Logistic Regression") and (row["Feature set"] == "TF-IDF + features")
            return ['background-color:#dcfce7; font-weight:600' if best else '' for _ in row]
        sty = (disp.style.apply(hl, axis=1)
               .format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}))
        st.dataframe(sty, use_container_width=True, hide_index=True)
        st.caption("Adding the linguistic features lifts the fake-class F1 from 0.532 (text only) to 0.727, "
                   "with sentiment the strongest single indicator of a fake review.")
