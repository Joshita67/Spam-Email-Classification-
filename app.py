import streamlit as st
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.title("📩 Spam Message Classifier")

msg = st.text_area("Enter your message:")

if st.button("Predict"):
    msg_clean = preprocess(msg)
    vect_msg = vectorizer.transform([msg_clean])
    prediction = model.predict(vect_msg)[0]
    label = "Spam" if prediction == 1 else "Ham"
    st.success(f"### Prediction: {label}")
