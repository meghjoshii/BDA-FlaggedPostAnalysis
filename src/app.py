from flask import Flask, render_template, url_for, request
import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import text
from sklearn.model_selection import train_test_split
from helper import *
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
import pandas as pd

from collections import Counter
from scipy import stats
import spacy
from textstat.textstat import textstatistics, legacy_round
import random
from sklearn.metrics import classification_report
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import load_model


#Set Data Directories

DATA_DIR_path = os.path.join(str(os.getcwd()),"data")
MODEL_DIR_path = os.path.join(str(os.getcwd()),"model")

file_path = os.path.join(DATA_DIR_path, "data.csv")
model_path = os.path.join(MODEL_DIR_path,"model.h5")


def initiate_model():
    """
    Loads the model weights to the software
    """

    #Preprocess Data
    data = pd.read_csv(file_path)
    data['Text'] = data.Body.apply(lambda x: BeautifulSoup(x, 'html.parser').text)
    data['Text'] = data['Text'].str.lower()
    #Set Parameters
    max_length = 200
    MAX_FEATURES = 20000
    MAX_LEN = 200
    X = data['Text'].values
    #Tokenise the data
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    #Get the indexes
    word_index = tokenizer.word_index
    #Initiate Label Encoder
    encoder = LabelEncoder()
    encoder.fit(data.Y.values)
    #Train and Validation splits
    train, validation = train_test_split(data, test_size=0.25, random_state=55)
    encoded_Y_train = encoder.transform(train.Y.values)
    encoded_Y_valid = encoder.transform(validation.Y.values)

    x_train = train.Text.values
    x_valid = validation.Text.values
    #Convert to categorical
    y_train = np_utils.to_categorical(encoded_Y_train)
    y_valid = np_utils.to_categorical(encoded_Y_valid)
    #Get tokens
    tokens = text.Tokenizer(num_words=MAX_FEATURES, lower=True)
    tokens.fit_on_texts(list(x_train))
    x_train = tokens.texts_to_sequences(x_train)
    x_valid = tokens.texts_to_sequences(x_valid)
    #Pad sequences
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_valid = tf.keras.preprocessing.sequence.pad_sequences(x_valid, maxlen=MAX_LEN)
    #Start the KERAS functional API
    inputs = tf.keras.Input(shape=(None,), dtype="int32")
    x = layers.Embedding(MAX_FEATURES, 128)(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    # x = layers.Flatten()(x)
    # x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    #model.summary()
    #Set the optimizers
    SGD = tf.keras.optimizers.SGD(learning_rate=0.01)
    #Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=[tf.keras.metrics.AUC()])
    model = load_model(model_path)
    return model




app = Flask(__name__,static_folder="static")
#Set Images confihuration
IMAGE_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
#Load the model
model = initiate_model()


@app.route('/')
def home():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
    return render_template('home.html', user_image = full_filename)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        content = {}
        """
        Get the post body for processing
        """
        review = request.form['review']
        d = [review]
        seq = tokenizer.texts_to_sequences(d)
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
        """
        Calculate featurws on the payload received
        """
        content['Average_Term_Entropy'] = str(round(avg_term_entropy(str(review)),4))
        content['Automated_Reading_Index'] = str(round(automated_reading_index(str(review)),4))
        content['Coleman_Liau_Index'] = str(round(coleman_liau_index(str(review)),4))
        content['Flesch_Kincaid_Grade_Level'] = str(round(flesch_kincaid_grade_level(str(review)), 4))
        content['Flesch_Reading_Ease_Score'] = str(round(flesch_reading_ease_score(str(review)), 4))
        content['Gunning_Fox_Index'] = str(round(gunning_fox_index(str(review)), 4))
        content['Smog_Index'] = str(round(smog_index(str(review)), 4))
        content['word_count'] = str(round(word_count(str(review)), 4))
        content['sentence_count'] = str(round(sentence_count(str(review)), 4))
        content['syllables_count'] = str(round(syllables_count(str(review)), 4))
        content['poly_syllable_count'] = str(round(poly_syllable_count(str(review)), 4))
        content['avg_sentences'] = str(round(avgSentences(str(review)), 4))
        content['avg_letters'] = str(round(avgLetters(str(review)), 4))
        """
        Perform Prediction
        """
        pred = model.predict(padded)
        labels = list(encoder.classes_)
        print(np.argmax(pred), labels[np.argmax(pred)])
        
    """
    Return logo and the prediction label
    """
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
    return render_template('result.html', prediction=labels[np.argmax(pred)],user_image = full_filename,**content)


if __name__ == '__main__':
    app.run(debug=True)