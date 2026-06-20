%%writefile app.py

import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import re
import base64
import os

# 1. CONFIG
st.set_page_config(
    page_title="Fake News Checker",
    page_icon="🪄",
    layout="wide"
)

# 2. LOAD LOGO
def get_base64_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        return ""
    except Exception as e:
        st.warning(f"Logo loading error: {e}")
        return ""

# Change path if needed
img_base64 = get_base64_image("sticker.png")

# 3. CUSTOM CSS
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #1b1338, #24164d, #2d1b69);
    color: #ffffff;
}

/* REMOVE DEFAULT STREAMLIT WHITE SPACE */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* LOGO CONTAINER */
.logo-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    margin-bottom: 10px;
}

/* LOGO IMAGE */
.logo-img {
    width: 75px;
    height: 75px;
    border-radius: 20px;
    object-fit: cover;
    box-shadow: 0 0 20px rgba(192,132,252,0.35);
}

/* LOGO TEXT */
.logo-text {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #c4b5fd, #f9a8d4, #fdba74);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUBTITLE */
.subtext {
    text-align: center;
    color: #cbd5e1;
    font-size: 16px;
    margin-bottom: 35px;
}

/* GLASS CARD */
.card {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(255,255,255,0.10);
    padding: 22px;
    border-radius: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

/* SECTION HEADER */
.section-header {
    font-size: 13px;
    font-weight: 700;
    color: #d8b4fe;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 14px;
    border-left: 4px solid #c084fc;
    padding-left: 10px;
}

/* INPUTS */
div[data-baseweb="input"] input {
    background-color: #0f172a !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
}

div[data-baseweb="textarea"] textarea {
    background-color: #0f172a !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
}

/* PLACEHOLDER */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #94a3b8 !important;
}

/* TEXTAREA */
textarea {
    background-color: #0f172a !important;
    color: white !important;
}

/* INPUT FOCUS */
.stTextInput input:focus,
.stTextArea textarea:focus {
    border: 1px solid #a855f7 !important;
    box-shadow: 0 0 12px rgba(168,85,247,0.5) !important;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #8b5cf6, #c084fc);
    color: white;
    border: none;
    padding: 14px;
    border-radius: 16px;
    font-weight: bold;
    font-size: 15px;
    transition: 0.3s ease;
    margin-top: 10px;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 20px rgba(192,132,252,0.5);
}

/* METRICS */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 18px;
}

/* TEXT COLORS */
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

/* INFO BOX */
.stAlert {
    background-color: rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
}

/* CONFIDENCE TEXT */
.confidence-text {
    font-size: 15px;
    font-weight: 500;
    color: #cbd5e1;
    margin-top: -10px;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #7c3aed;
    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# 4. LOAD MODELS
@st.cache_resource
def load_all():
    try:
        model_data = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model_data, vectorizer
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

data, tfidf = load_all()

# 5. HEADER
st.markdown(f"""
<div class="logo-container">
    <img src="data:image/png;base64,{img_base64}" class="logo-img">
    <span class="logo-text">Fake News Integrity Checker</span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="subtext">
        AI-powered fake news detection, sentiment analysis & explainable insights.
    </div>
    """,
    unsafe_allow_html=True
)

# 6. LAYOUT
col1, col2 = st.columns([1, 1.3], gap="large")

# LEFT SIDE
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<p class="section-header">1. Content Input</p>',
        unsafe_allow_html=True
    )

    headline = st.text_input("Headline (Optional)")

    txt = st.text_area(
        "Paste news article text here...",
        height=320
    )

    btn = st.button("RUN DEEP ANALYSIS")

    st.markdown('</div>', unsafe_allow_html=True)

    # WORD IMPORTANCE
    if btn and txt and tfidf is not None:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(
            '<p class="section-header">4. Word Importance (Explainable AI)</p>',
            unsafe_allow_html=True
        )

        try:
            features = tfidf.get_feature_names_out()
            matrix = tfidf.transform([txt]).toarray()[0]

            top_indices = matrix.argsort()[-10:][::-1]

            impact_data = [
                {"Word": features[i], "Weight": matrix[i]}
                for i in top_indices
                if matrix[i] > 0
            ]

            if impact_data:

                df_imp = pd.DataFrame(impact_data).sort_values(
                    by="Weight",
                    ascending=True
                )

                fig_bar = px.bar(
                    df_imp,
                    x="Weight",
                    y="Word",
                    orientation='h',
                    color='Weight',
                    color_continuous_scale='Purples'
                )

                fig_bar.update_layout(
                    height=340,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    margin=dict(t=0, b=0, l=0, r=10),
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"Word importance error: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

# RIGHT SIDE
with col2:

    st.markdown(
        '<p class="section-header">Analysis & Insights</p>',
        unsafe_allow_html=True
    )

    if btn:

        if not txt.strip():
            st.warning("Please enter article text.")
        elif data is None or tfidf is None:
            st.error("Model or vectorizer not found.")
        else:

            try:
                # MODEL PREDICTION
                vec = tfidf.transform([txt])

                p_real = (
                    0.4 * data["svm"].predict_proba(vec)[0][1] +
                    0.6 * data["xgb"].predict_proba(vec)[0][1]
                )

                # LOGIC
                if p_real >= 0.5:
                    label = "REAL"
                    conf = int(p_real * 100)
                    color = "#16a34a"
                else:
                    label = "FAKE"
                    conf = int((1 - p_real) * 100)
                    color = "#dc2626"

                # CONFIDENCE LEVEL
                if conf >= 90:
                    level_text = "High Confidence"
                elif conf >= 70:
                    level_text = "Medium Confidence"
                else:
                    level_text = "Low Confidence"

                # VERDICT CARD
                st.markdown(f"""
                <div class="card">
                    <p class="section-header">Verdict</p>
                    <h2 style="color:{color}; margin-bottom:5px;">
                        {conf}% CONFIDENCE: {label}
                    </h2>
                    <p class="confidence-text">
                        Status: {level_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # GAUGE CHART
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf,
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color}
                    }
                ))

                fig_g.update_layout(
                    height=220,
                    margin=dict(t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': "white"}
                )

                st.plotly_chart(fig_g, use_container_width=True)

                # SENTIMENT + STYLE
                c_a, c_b = st.columns(2)

                with c_a:

                    pol = TextBlob(txt).sentiment.polarity

                    st.markdown(f"""
                    <div class="card">
                        <p class="section-header">Sentiment</p>
                        <h3>{pol:.2f}</h3>
                        <p style="font-size:11px;">
                            Emotional Tone
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with c_b:

                    words = txt.split()

                    caps = len([
                        w for w in words if w.isupper()
                    ]) / (len(words) + 1)

                    st.markdown(f"""
                    <div class="card">
                        <p class="section-header">Style</p>
                        <h3>{caps:.1%}</h3>
                        <p style="font-size:11px;">
                            Caps Lock Ratio
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # TEXT STATISTICS
                st.markdown('<div class="card">', unsafe_allow_html=True)

                st.markdown(
                    '<p class="section-header">5. Text Statistics</p>',
                    unsafe_allow_html=True
                )

                s1, s2, s3 = st.columns(3)

                s1.metric("Word Count", len(words))
                s2.metric("Char Count", len(txt))
                s3.metric(
                    "Sentences",
                    max(1, len(re.findall(r'[.!?]+', txt)))
                )

                st.markdown('</div>', unsafe_allow_html=True)

                # WORDCLOUD
                st.markdown(
                    '<p class="section-header">6. Top Keywords Wordcloud</p>',
                    unsafe_allow_html=True
                )

                wc_scheme = "Reds" if label == "FAKE" else "Greens"

                wc = WordCloud(
                    width=800,
                    height=300,
                    background_color="white",
                    colormap=wc_scheme
                ).generate(txt.lower())

                st.image(
                    wc.to_array(),
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

    else:
        st.info("Input content and click analysis to start.")
