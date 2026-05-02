import numpy as np
import pandas as pd
df=pd.read_csv('spam.csv',encoding="latin1")
print(df.sample(5))
print(df.shape)
#1.Data Cleaning
#2.EDA
#3.Text Preprocessing
#4.Model Building
#5.Evaluation
#6.Improvements
#7.Website
#8.Deployment

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
print(df.sample)

#renaming the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
print(df.sample)

#encoding ham and spam
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
print(df.head())

#checking missing values
print(df.isnull().sum())

#check for duplicate values
print(df.duplicated().sum())
#remove duplicate values
df=df.drop_duplicates(keep='first')
print(df.shape)
print(df.value_counts(['target']))

#pie chart
import matplotlib.pyplot as plt
plt.pie(df.value_counts(['target']),labels=['ham','spam'],autopct="%0.2f")
plt.show()

import nltk    #natural language toolkit
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.data.path.append('C:\\Users\\Rudra\\AppData\\Roaming\\nltk_data')
df['num_characters']=df['text'].apply(len)   #finds the length of all the texts
print(df['num_characters'])
df['no_of_words']=df['text'].apply(lambda x: len(nltk.word_tokenize(x)))   #counts the number of words in a text
print(df['no_of_words'])

df['no_of_sentence']=df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
print(df['no_of_sentence'])

print(df.head())

print(df[['num_characters','no_of_words','no_of_sentence']].describe())

import seaborn as sns
#plotting histogram for spam and ham msgs
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')
plt.show()
sns.pairplot(df,hue='target')
plt.show()


#data preprocessing ->  lower case, tokenization, removing special characters, removing stop words and punctuations, stemming
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')
from nltk.stem.porter import PorterStemmer
nltk.data.path.append('C:\\Users\\Rudra\\AppData\\Roaming\\nltk_data')
import string
string.punctuation

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


df['transformed_text']=df['text'].apply(transform_text)
print(df.head(5))

#counting the top 30 ham and spam keywords
ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
from collections import Counter
print(Counter(ham_corpus).most_common(30))

spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
from collections import Counter
print(Counter(spam_corpus).most_common(30))


#MODEL BUILDING 
#for ML we need to use numerical data to build models, so we use vectorization to change transformed_text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
X = tfidf.fit_transform(df['text']).toarray()
y=df['target'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
# we are using naive bayes for this because naive bayes works better in textual data
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.preprocessing import MinMaxScaler  #we using minmax scaler and not standard scaler because standard scaler gives negative values which are not compatible with naive bayes
scaler=MinMaxScaler()
X=scaler.fit_transform(X)

gnb=GaussianNB()
bnb=BernoulliNB()
mnb=MultinomialNB()

gnb.fit(X_train,y_train)
y_pred1=gnb.predict(X_test)
print("accuracy_score: ",accuracy_score(y_test,y_pred1))
print("confusion_matrix: ",confusion_matrix(y_test,y_pred1))
print("precision_score: ",precision_score(y_test,y_pred1))

bnb.fit(X_train,y_train)
y_pred2=bnb.predict(X_test)
print("accuracy_score: ",accuracy_score(y_test,y_pred2))
print("confusion_matrix: ",confusion_matrix(y_test,y_pred2))
print("precision_score: ",precision_score(y_test,y_pred2))


#here, we will use mnb because the precision score is better than the others(100%)
mnb.fit(X_train,y_train)
y_pred3=mnb.predict(X_test)
print("accuracy_score: ",accuracy_score(y_test,y_pred3))
print("confusion_matrix: ",confusion_matrix(y_test,y_pred3))
print("precision_score: ",precision_score(y_test,y_pred3))


import pickle
#save model
pickle.dump(mnb,open('model.pkl','wb'))
#save vectorization
pickle.dump(tfidf,open('vectorization.pkl','wb'))



