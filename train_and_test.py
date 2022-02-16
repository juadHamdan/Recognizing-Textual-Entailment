"""
Description: Text classification on the SNLI dataset using pre-trained GloVe word embeddings.
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

TrainFileName = "snli_1.0_train.jsonl"
TestFileName = "snli_1.0_test.jsonl"
gloveFileName = "glove.6B.300d.txt"
embedding_dim = 300

max_seq_length = 20  # maximum length of all sequences

three_way_num_classes = 3
two_way_num_classes = 2

Entailment = 'entailment'
NonEntailment = 'non-entailment'
Neutral = 'neutral'
Contradiction = 'contradiction'

non_entailment_labels = [Neutral, Contradiction]
two_way_label_category = {Entailment: 0, NonEntailment: 1}
three_way_label_category = {Neutral: 0, Entailment: 1, Contradiction: 2}

"""
## Introduction
Train a text classification model that uses pre-trained word embeddings.
We'll work with the SNLI corpus (https://nlp.stanford.edu/projects/snli/),
For the pre-trained word embeddings, we'll use GloVe embeddings (http://nlp.stanford.edu/projects/glove/).
"""


def LoadThreeWayClassificationDataset(filename):
    texts = []
    hypotheses = []
    labels = []

    with open(filename, 'r') as f:
        for line in f:
            row = json.loads(line)
            label = row['gold_label'].strip()
            if label in three_way_label_category:
                texts.append(row['sentence1'].encode('utf8').strip())
                hypotheses.append(row['sentence2'].encode('utf8').strip())
                labels.append(three_way_label_category[label])

    return texts, hypotheses, labels


def LoadTwoWayClassificationDataset(filename):
    texts = []
    hypotheses = []
    labels = []

    with open(filename, 'r') as f:
        for line in f:
            row = json.loads(line)
            label = row['gold_label'].strip()
            if label in three_way_label_category:
                texts.append(row['sentence1'].encode('utf8').strip())
                hypotheses.append(row['sentence2'].encode('utf8').strip())
                if label in non_entailment_labels:
                    labels.append(two_way_label_category[NonEntailment])
                else:
                    labels.append(two_way_label_category[Entailment])

    return texts, hypotheses, labels


def LoadPreTrainedWordEmbeddings():
    # make a dictionary mapping words to their NumPy vector representation
    embeddings = {}

    with open(gloveFileName, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding

    return embeddings


def PrepareEmbeddingMatrix(embeddings, NUM_WORDS, word_index):
    words_len = min(NUM_WORDS, len(word_index))
    # word_embedding_matrix = np.zeros((words_len + 1, embedding_dim))
    word_embedding_matrix = np.random.random((words_len + 1, embedding_dim))
    k = 0
    for word, i in word_index.items():
        if i >= NUM_WORDS:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
            k += 1

    return word_embedding_matrix, words_len


def Embed(word_embedding_matrix, words_len, input):
    # Note that we set `trainable=False` so as to keep the embeddings fixed.
    # (we don't want to update them during training).
    return layers.Embedding(
        words_len + 1,
        embedding_dim,
        input_length=max_seq_length,
        embeddings_initializer=keras.initializers.Constant(word_embedding_matrix),
        trainable=False,
    )(input)


def Preprocess(tokenizer, texts, hypotheses, labels):
    # Convert our list-of-strings data to NumPy padded arrays of integer indices.
    # texts_to_sequences: Transforms each text in texts to a sequence of integers.
    text_word_sequences = tokenizer.texts_to_sequences([x.decode('utf-8') for x in texts])
    hypotheses_word_sequences = tokenizer.texts_to_sequences([x.decode('utf-8') for x in hypotheses])
    x_text = tf.keras.preprocessing.sequence.pad_sequences(text_word_sequences, maxlen=max_seq_length)
    x_hypo = tf.keras.preprocessing.sequence.pad_sequences(hypotheses_word_sequences, maxlen=max_seq_length)
    # to_categorical: converts a class vector (integers) to binary class matrix.
    y = tf.keras.utils.to_categorical(np.asarray(labels))
    print('Shape of text tensor:', x_text.shape)
    print('Shape of hypotheses tensor:', x_hypo.shape)
    print('Shape of label tensor:', y.shape)

    return x_text, x_hypo, y


def LoadModelAndTokenizer(modelName):
    model = keras.models.load_model(modelName)
    with open(modelName + ' tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return model, tokenizer


def SaveModelAndTokenizer(model, tokenizer, modelName):
    model.save(modelName)
    with open(modelName + ' tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def TestModel(model, tokenizer, LoadDatasetFunction):
    test_text, test_hypotheses, test_labels = LoadDatasetFunction(TestFileName)
    x_text_val, x_hypo_val, y_val = Preprocess(tokenizer, test_text, test_hypotheses, test_labels)
    _, accuracy = model.evaluate([x_text_val, x_hypo_val], y_val)
    return accuracy


def TrainModel(model, inputs, labels):
    model.fit(inputs,
              labels,
              epochs=1,
              validation_split=0.1,
              shuffle=False,  # True,
              verbose=2) # validation_data=([x_text_val, x_hypo_val], y_val))


def CreateVocabularyIndex(num_words, train_sentences):
    # This class allows to tokenize a text corpus, by turning each text into a sequence of integers
    # (each integer being the index of a token in a dictionary)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)
    # fit_on_texts: Updates internal vocabulary based on a list of texts.
    tokenizer.fit_on_texts([x.decode('utf-8') for x in train_sentences])
    word_index_vocabulary = tokenizer.word_index  # also num of unique tokens

    return tokenizer, word_index_vocabulary


def InstantiateInputs():
    text_input = tf.keras.Input(shape=(max_seq_length,), dtype='int32', name='text')
    hypo_input = tf.keras.Input(shape=(max_seq_length,), dtype='int32', name='hypothesis')

    return text_input, hypo_input


def LoadWordEmbeddingsIntoEmbeddingLayer(word_embedding_matrix, words_len, text_input, hypo_input):
    """
    Load the pre-trained word embeddings matrix into an Embedding layer.
    """
    text_features = Embed(word_embedding_matrix, words_len, text_input)
    hypo_features = Embed(word_embedding_matrix, words_len, hypo_input)

    return text_features, hypo_features


def BuildModel(text_input, hypo_input, text_features, hypo_features, num_classes):
    # Reduce sequence of embedded words in the title into a single 20-dimensional vector
    text_features = layers.LSTM(max_seq_length)(text_features)
    # Reduce sequence of embedded words in the body into a single 20-dimensional vector
    hypo_features = layers.LSTM(max_seq_length)(hypo_features)
    # Merge all available features into a single large vector via concatenation
    x = layers.concatenate([text_features, hypo_features])
    # Stick a logistic regression for priority prediction on top of the features
    label_pred = layers.Dense(num_classes, name="label")(x)
    # Instantiate an end-to-end model predicting both priority and department
    model = keras.Model(inputs=[text_input, hypo_input], outputs=[label_pred])
    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

    return model


def Classification(num_classes, LoadDatasetFunction, modelName):
    train_text, train_hypotheses, train_labels = LoadDatasetFunction(TrainFileName)
    embeddings = LoadPreTrainedWordEmbeddings()
    NUM_WORDS = len(embeddings)

    tokenizer, word_index_vocabulary = CreateVocabularyIndex(len(embeddings), train_text + train_hypotheses)
    text_data, hypo_data, labels = Preprocess(tokenizer, train_text, train_hypotheses, train_labels)
    word_embedding_matrix, words_len = PrepareEmbeddingMatrix(embeddings, NUM_WORDS, word_index_vocabulary)
    text_input, hypo_input = InstantiateInputs()
    text_features, hypo_features = LoadWordEmbeddingsIntoEmbeddingLayer(word_embedding_matrix, words_len, text_input, hypo_input)

    model = BuildModel(text_input, hypo_input, text_features, hypo_features, num_classes)
    TrainModel(model, [text_data, hypo_data], labels)
    accuracy = TestModel(model, tokenizer, LoadDatasetFunction)
    # SaveModelAndTokenizer(model, tokenizer, modelName)

    return accuracy


if __name__ == "__main__":
    two_way_accuracy = Classification(two_way_num_classes, LoadTwoWayClassificationDataset, '2_way_model')
    print("2-way classification accuracy: ", two_way_accuracy)

    # three_way_accuracy = Classification(three_way_num_classes, LoadThreeWayClassificationDataset, '3_way_model')
    # print("3-way classification accuracy: ", three_way_accuracy)
