import pandas as pd
import numpy as np
import seaborn as sns
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import streamlit as st
import pickle

@st.cache

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model.load_model('model.h5')

def predict(text):
    txts = tok.texts_to_sequences(text)
    txts = pad_sequences(txts, maxlen=max_len)
    preds = model.predict(txts)
    return preds

st.title('Spam or Ham')
st.header('Enter any string:')
max_words = 1000
max_len = 150
sample_texts = st.text_input('sample_texts')
with open('tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)

if st.button('Predict Spam/Ham'):
    price = predict(sample_texts)
    if(price>0.5):
        st.success('Spam')
    else:
        st.success('Ham')
