import streamlit as st
import json
import unidecode
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def normalize(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.4:
        return "pozytywny"
    elif scores['compound'] <= -0.4:
        return "negatywny"
    else:
        return "neutralny"

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

with open("faq.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

st.title("Spory o Zasady Gry – Grzybobranie Rodzina Treflików")

user_q = st.text_input("Opisz swój spór lub pytanie:")

if user_q:
    mood = analyze_sentiment(user_q)
    answer = search_answer(user_q, faq)
    
    # Komunikaty sentymentowe lub informacyjne
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
