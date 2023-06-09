import os
import pickle
import pyttsx3
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

current_dir = os.path.dirname(os.path.realpath(__file__))
model = load_model(os.path.join(current_dir, 'models', 'textcnnMultiChannel.h5'))

path_tok = os.path.join(current_dir, 'models', 'tokenizer.h5')
with open(path_tok, 'rb') as f:
    tokenizer = pickle.load(f)

st.title("Polarity Detection with AI!")

text = st.text_area("Enter Your Text: ", height=200)

btn = st.button("Detect Polarity")

if btn:
    encoded_doc = tokenizer.texts_to_sequences([text])
    padded_doc = pad_sequences(encoded_doc, maxlen=1693, padding='post')
    predict = model.predict([padded_doc, padded_doc, padded_doc, padded_doc])[0][0]
    if predict < 0.5:
        engine = pyttsx3.init()
        engine.say("Polarity is Negative")
        st.error('Polarity is: Negative 🤬')
        engine.runAndWait()
    else:
        st.success("Polarity is: Positive 😍")
        engine = pyttsx3.init()
        engine.say("Polarity is Positive")
        engine.runAndWait()