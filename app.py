import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load model and vectorizer
with open('spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Page config
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱")

st.title("📱 SMS Spam Classifier")
st.write("Built by **Priyanshu Mewa** | B.Tech ECE, IIIT Una")
st.markdown("---")

st.subheader("Enter an SMS message to check if it's spam:")
user_input = st.text_area("Type your message here", height=150,
                           placeholder="e.g. Congratulations! You won a free prize...")

if st.button("🔍 Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        cleaned = preprocess_text(user_input)
        vectorised = tfidf.transform([cleaned])
        prediction = model.predict(vectorised)[0]
        probability = model.predict_proba(vectorised)[0]
        confidence = max(probability) * 100

        if prediction == 1:
            st.error(f"🚨 SPAM — Confidence: {confidence:.1f}%")
        else:
            st.success(f"✅ HAM (Not Spam) — Confidence: {confidence:.1f}%")

st.markdown("---")
st.markdown("### 📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "96.77%")
col2.metric("Spam Precision", "100%")
col3.metric("Weighted F1", "0.97")

st.caption("Model: Multinomial Naive Bayes | Dataset: UCI SMS Spam Collection (5,572 messages)")