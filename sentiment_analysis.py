#!/usr/bin/env python
# coding: utf-8


#Used CONTRACTIONS library
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# ! pip install bs4 # in case you don't have it installed
# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# Read Data
dataset=pd.read_csv('./data.tsv',header=0,on_bad_lines='skip',sep='\t') 


# Keep Reviews and Ratings

df=dataset.iloc[:,[7,13]]

df['star_rating']=df['star_rating'].apply(lambda x:1 if x in [1,2] else (2 if x==3 else 3))


data=df.groupby('star_rating').sample(n=20000)



# Data Cleaning


len_1=data['review_body'].str.len()
avg_len_1=len_1.mean()


#remove lower case
data['review_body']=data['review_body'].str.lower()

#remove extra space
data['review_body']=data['review_body'].str.strip()


#remove URL
data['review_body']=data['review_body'].apply(lambda x:re.split('https:\/\/.*',str(x))[0])

#remove HTML texts
data['review_body']=data['review_body'].str.replace(r'<[^<>]*>','',regex=True)

#using contractions  
# ! pip install contractions
data['review_body']=data['review_body'].apply(lambda x: contractions.fix(x))

#remove non-alphabetical characters
data['review_body'] = data['review_body'].apply(lambda x: re.sub(r'[^a-z\s]+', ' ', x))

#again remove extra space
data['review_body'] = data['review_body'].str.strip()

#drop null values
data=data.dropna()


len_2=data['review_body'].str.len()
avg_len_2=len_2.mean()


# pre-processing

length_1=data['review_body'].str.len()
avg_length_1=length_1.mean()


# remove the stop words 

stop = stopwords.words('english')
data['new']= data['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# perform lemmatization  

lemmatizer=WordNetLemmatizer()

# Machinelearningplus.com. (2018). [online] Available at: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/ [Accessed 25 Jan. 2023].

def get_wordnet_pos(word):
    tag=nltk.pos_tag([word])[0][1][0].upper()
    tag_dict={"J": wordnet.ADJ,"N": wordnet.NOUN,"V": wordnet.VERB,"R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in nltk.word_tokenize(text)])

data['new'] = data.new.apply(lemmatize_text)

# Stack Overflow. (2017). Lemmatization of all pandas cells. [online] Available at: https://stackoverflow.com/questions/47557563/lemmatization-of-all-pandas-cells [Accessed 19 Jan. 2023].
length_2=data['new'].str.len()
avg_length_2=length_2.mean()


# TF-IDF Feature Extraction

v=TfidfVectorizer(ngram_range=(1,3))

x=v.fit_transform(data['new'])

df_new=pd.DataFrame(data=x.toarray(),columns=v.get_feature_names_out())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,data['star_rating'],test_size=0.2,random_state=42,shuffle="true")


# Perceptron

p=Perceptron()
p.fit(X_train,y_train)
y_pred_perc=p.predict(X_test)

precision_perc,recall_perc,f1score_perc,support_perc=score(y_test, y_pred_perc)

precision_perc_avg,recall_perc_avg,f1score_perc_avg,support_perc_avg=score(y_test, y_pred_perc, average='weighted')


# SVM

svm=LinearSVC()
svm.fit(X_train, y_train)
y_pred_svm=svm.predict(X_test)

precision_svm,recall_svm,f1score_svm,support_svm=score(y_test, y_pred_svm)

precision_svm_avg,recall_svm_avg,f1score_svm_avg,support_svm_avg=score(y_test, y_pred_svm, average='weighted')


# Logistic Regression

lr=LogisticRegression()
lr.fit(X_train, y_train)
y_pred_logreg=lr.predict(X_test)

precision_logreg,recall_logreg,f1score_logreg,support_logreg=score(y_test, y_pred_logreg)

precision_logreg_avg,recall_logreg_avg,f1score_logreg_avg,support_logreg_avg=score(y_test, y_pred_logreg, average='weighted')


# Naive Bayes

nb=MultinomialNB()
nb.fit(X_train, y_train)
y_pred_NB=nb.predict(X_test)

precision_nb,recall_nb,f1score_nb,support_nb=score(y_test, y_pred_NB)

precision_nb_avg,recall_nb_avg,f1score_nb_avg,support_nb_avg=score(y_test, y_pred_NB, average='weighted')


# Printing

print("Average length of reviews before and after data cleaning ",avg_len_1, ',', avg_len_2)

print("Average length of reviews before and after data preprocessing ",avg_length_1, ',', avg_length_2)


print("Perceptron")
a=0
for i,j,k in zip(precision_perc, recall_perc, f1score_perc):
    a+=1
    print('Class',a,': Precision, Recall, F1 Score - ' ,i,',',j,',',k) 
print('Average : Precision, Recall, F1 Score - ',precision_perc_avg,',',recall_perc_avg,',',f1score_perc_avg)


print("SVM")
b=0
for i,j,k in zip(precision_svm, recall_svm, f1score_svm):
    b+=1
    print('Class',b,': Precision, Recall, F1 Score - ' ,i,',',j,',',k )
print('Average : Precision, Recall, F1 Score - ',precision_svm_avg,',',recall_svm_avg,',',f1score_svm_avg)


print("Logistic Regression")
c=0
for i,j,k in zip(precision_logreg, recall_logreg, f1score_logreg):
    c+=1
    print('Class',c,': Precision, Recall, F1 Score - ' ,i,',',j,',',k )
print('Average : Precision, Recall, F1 Score - ',precision_logreg_avg,',',recall_logreg_avg,',',f1score_logreg_avg)


print("Naive Bayes")
d=0
for i,j,k in zip(precision_nb, recall_nb, f1score_nb):
    d+=1
    print('Class',d,': Precision, Recall, F1 Score - ' ,i,',',j,',',k )
print('Average : Precision, Recall, F1 Score - ',precision_nb_avg,',',recall_nb_avg,',',f1score_nb_avg)




