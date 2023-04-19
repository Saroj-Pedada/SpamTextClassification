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
from keras.models import load_model
import streamlit as st
import pickle

max_words = 1000
max_len = 150
model = load_model('model.h5')

st.title('Spam or Ham')

sample_texts = st.text_input('Spam/Ham Text Classifier')

with open('tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)

txts = tok.texts_to_sequences([sample_texts])
txts = pad_sequences(txts, maxlen=max_len)
preds = model.predict(txts)
print(preds)
if st.button('Predict'):
    if(preds<=0.2):
        st.success("Ham")
    else:
        st.success("Spam")
