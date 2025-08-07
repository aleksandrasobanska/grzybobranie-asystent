import streamlit as st
import json
import unidecode
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Normalizacja tekstu
def normalize(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Wczytanie modelu do analizy sentymentu
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("allegro/herbert-base-sentiment")
    return tokenizer, model

tokenizer, model = load_sentiment_model()

# Analiza sentymentu w języku polskim
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = softmax(logits.numpy()[0])
    labels = ["negatywny", "neutralny", "pozytywny"]
    return labels[scores.argmax()]

# Dopasowywanie odpowiedzi z bazy FAQ
def search_answer(question, faq_base):
    q_words = set(normalize(question).split())
    max_found = 0
    best_answer = None
    for item in faq_base:
        for kw in item["keywords"]:
            k_words = set(normalize(kw).split())
            common = len(q_words & k_words)
            if common > max_found and common > 0:
                max_found = common
                best_answer = item["answer"]
    return best_answer

# Wczytanie bazy FAQ
with open("faq.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

# Interfejs Streamlit
st.title("Spory o Zasady Gry – Grzybobranie Rodzina Treflików")

user_q = st.text_input("Opisz swój spór lub pytanie:")

if user_q:
    mood = analyze_sentiment(user_q)
    answer = search_answer(user_q, faq)

    # Komunikaty sentymentowe
    message_shown = False
    if mood == "negatywny":
        st.markdown("😟 **Wygląda na to, że jesteś sfrustrowany/a. Postaram się pomóc jak najlepiej!**")
        message_shown = True
    elif mood == "pozytywny":
        st.markdown("😊 **Super nastawienie! Zaraz znajdę odpowiedź!**")
        message_shown = True

    if answer:
        st.markdown(f"**Odpowiedź:** {answer}")
    else:
        if not message_shown:
            st.markdown("🔎 **Analizuję Twoje pytanie...**")
        st.markdown("❔ _Nie znalazłem jednoznacznej odpowiedzi. Opisz sytuację jeszcze dokładniej lub zadaj pytanie inaczej._")

st.info("Przykład: 'Co zrobić, gdy dwa pionki są na jednym polu?' lub 'Czy trzeba wyrzucić dokładnie tyle, by wejść na metę?'")
