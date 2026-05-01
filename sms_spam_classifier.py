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
#plt.show()

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
#plt.show()
sns.pairplot(df,hue='target')
#plt.show()
sns.heatmap(df.corr(),annot=True)

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
