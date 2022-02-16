import pickle
import tensorflow as tf
from tensorflow import keras

max_seq_length = 20


def PreprocessTextsAndHypotheses(tokenizer, texts, hypotheses):
    # Convert our list-of-strings data to NumPy padded arrays of integer indices.
    text_word_sequences = tokenizer.texts_to_sequences([texts])
    hypotheses_word_sequences = tokenizer.texts_to_sequences([hypotheses])
    x_text = tf.keras.preprocessing.sequence.pad_sequences(text_word_sequences, maxlen=max_seq_length)
    x_hypo = tf.keras.preprocessing.sequence.pad_sequences(hypotheses_word_sequences, maxlen=max_seq_length)
    return x_text, x_hypo


def LoadModelAndTokenizer(modelName):
    model = keras.models.load_model(modelName)
    with open(modelName + ' tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return model, tokenizer
