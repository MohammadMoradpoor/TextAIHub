import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random
import os
from hazm import word_tokenize
from hazm.Stemmer import Stemmer
stemmer = Stemmer()

current_dir = os.path.dirname(os.path.realpath(__file__))
model = os.path.join(current_dir, 'models', 'lion_v.jsdh')
lion_v = open(model, 'rb')
v = pickle.load(lion_v)
lion_v.close()

model = os.path.join(current_dir, 'models', 'lion_le.jsdh')
lion_le = open(model, 'rb')
le = pickle.load(lion_le)
lion_le.close()

model = os.path.join(current_dir, 'models', 'lion_svc.jsdh')
lion_svc = open(model, 'rb')
svc = pickle.load(lion_svc)
lion_svc.close()

model = os.path.join(current_dir, 'models', 'stopwords.txt')
with open(model, encoding='utf8') as stopwords_file:
    stopwords = stopwords_file.readlines()
stopwords = [str(line).replace('\n', '') for line in stopwords]

nltk_stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(nltk_stopwords)

st.title("News Category Detection with AI!")

text = st.text_area("Enter Your News Text: ")

btn = st.button("Detect Category")

if btn:
    title_body_tokenized = word_tokenize(text)
    title_body_tokenized_filtered = [w for w in title_body_tokenized if not w in stopwords]
    title_body_tokenized_filtered_stemmed = [stemmer.stem(w) for w in title_body_tokenized_filtered]
    x = [' '.join(title_body_tokenized_filtered_stemmed)]
    x_v = v.transform(x)
    p = svc.predict(x_v)
    label = le.inverse_transform(p)
    st.success("The Label is: " + str(label[0]))