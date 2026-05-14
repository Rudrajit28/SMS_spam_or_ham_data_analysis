import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="shield",
    layout="centered",
)

@st.cache_resource
def load_model():
    tfidf = pickle.load(open('spam_classifier/vectorization.pkl', 'rb'))
    model = pickle.load(open('spam_classifier/model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_model()

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)


# Header
st.title("SMS Spam Detector")
st.caption("Powered by Multinomial Naive Bayes · TF-IDF Vectorizer")

st.divider()

# Stats
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "97%")
col2.metric("Precision", "100%")
col3.metric("Trained on", "5.5K+ msgs")

st.divider()

# Input
input_sms = st.text_area(
    "Paste your SMS message below",
    placeholder='e.g. "Congratulations! You\'ve won £1,000. Call now to claim..."',
    height=150,
)

predict_clicked = st.button("Analyse Message", type="primary", use_container_width=True)

# Result
if predict_clicked:
    if not input_sms.strip():
        st.warning("Please enter a message before analysing.")
    else:
        with st.spinner("Analysing message..."):
            transformed = transform_text(input_sms)
            vectorized  = tfidf.transform([transformed])
            result      = model.predict(vectorized)[0]

        st.divider()

        if result == 1:
            st.error("Spam Detected")
            st.error("This message shows patterns typical of spam. Do not click any links or share personal information.")
        else:
            st.success("Not Spam")
            st.success("This message appears legitimate. No suspicious patterns were detected.")

        with st.expander("See processed text"):
            st.code(transform_text(input_sms), language=None)
#to run the code write "streamlit run app.py" in the terminal
#working link--https://smsspamorhamdataanalysis12.streamlit.app/
