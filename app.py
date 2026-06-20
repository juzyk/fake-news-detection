import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Streamlit UI
st.title("Fake News Detection App")

st.write("""
This application predicts whether a news article is:
- REAL
- FAKE
""")

# Text input
user_input = st.text_area(
    "Enter news text here:",
    height=200
)

# Predict button
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vectorized_text = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(vectorized_text)[0]

        # Probability
        probability = model.predict_proba(vectorized_text)[0]

        # Display result
        if prediction == 1:
            st.success("Prediction: REAL NEWS")
            confidence = probability[prediction] * 100
            st.write(f"Confidence: {confidence:.2f}%")
        else:
            st.error("Prediction: FAKE NEWS")
            confidence = probability[prediction] * 100
            st.write(f"Confidence: {confidence:.2f}%")
