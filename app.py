import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

tfidf=pickle.load(open('vectorization.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("SMS SPAM CLASSIFIER")
input_sms=st.text_area("Enter the message")

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    ps=PorterStemmer()
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

result = None  

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vectorization = tfidf.transform([transformed_sms])
    result = model.predict(vectorization)[0]

if result is not None:
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam ")
