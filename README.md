# Recognizing Textual Entailment - Report
<hr/>

Given two text fragments, decide whether the meaning of one text is entailed 
(can be inferred) from another text.

<hr/>

### My approach to the task
My vision for this task was:<br/>
Words from the training corpus => 
numeric representation that can approximate meanings and inference from the words => 
use that for model training.

We can accomplish this task using word embeddings (Distributional Semantics).
I chose to word with GloVe word embeddings.
We also need a corresponding machine learning algorithm to predict the entailment between the text and the hypothesis, based on our training corpus.

<strong>Why word embeddings?</strong><br/>
Word embedding is a numeric vector input that represent a word and can approximate meaning.
(Every word in my vocabulary will have a unique vector associated with it)
How does they can approximate meaning:
	Taken from Wikipedia: https://en.wikipedia.org/wiki/Distributional_semantics
"The distributional hypothesis in linguistics is derived from the semantic theory of language usage,
 i.e. words that are used and occur in the same contexts tend to purport similar meanings."

Words that are more similar to each other are going to be used together more often. <br/>
For example: <br/>
The words "orange" and "banana" are going to be used more with the words "watermelon", "fruits" and "food" than they will with the word "ladder".


What is the difference between GloVe and word2vec: <br/>
Word2Vec leverage co-occurance within local context (neighboring words),
GloVe (Global Vectors) on the other hand is based on leveraging global word to word co-occurance counts leveraging the entire corpus.
Both of these models give similar results. 


<hr/>

### Project Description 
Text classification on the SNLI dataset using pre-trained GloVe word embeddings.

<strong>Introduction</strong><br/>
Train a text classification model that uses pre-trained word embeddings.
We'll work with the SNLI corpus (https://nlp.stanford.edu/projects/snli/).
For the pre-trained word embeddings, we'll use GloVe embedding (http://nlp.stanford.edu/projects/glove/).

<strong>Program flow:</strong><br/>
 - Load and process SLNI dataset (get texts, hypotheses, labels) 
 - Load and process word embeddings (a dictionary mapping words to their numpy vector representation)
 - Create a vocabulary index (Index the vocabulary found in the dataset)
 - Use the same layer to vectorize the dataset
 - Prepare embeddings matrix
 It's a numpy matrix where entry at index i is the pre-trained vector for the word of index i in our vocabulary.
- Load the pre-trained word embeddings matrix into an Embedding layer
The embedding layer map the inputs (integers) to the vectors found at the corresponding index in the embedding matrix, 
i.e. the vector [ 1, 2, 3 ] would be converted to [ embeddings[1], embeddings[2], embeddings[3] ].
- Build model
 - Train the model
 - Test the model
 - Save the model


#### Results:
Two-way classification: (1) entailment and (2) non-entailment (contradiction + neutral): 0.76 <br/>
Three-way classification: (1) entailment, (2) contradiction and (3) neutral: 0.641



<hr/>

#### Errors encountered with

<strong> - Dimension mismatch:  </strong><br/>
Vectors can be in different dimensions (lengths).
	I trained the model using the vector of dimension 300, if I try to apply vectors of different dimension - errors will occur. (Dimensions mis-match errors)
	Solution:
We should make sure to use the same dimensions throughout.

<strong> - We need to use the exact same preprocessing and configurations during model Training, Testing and Predicting as was used to create the training data for the word embeddings in the first place. </strong><br/>
	For example: the use of a different Tokenizer, we are going to end up with incompatible input.<br/>
Solution: <br/>
We should work in an order while programming and fully understand the consequences.
	
<strong> - Out of vocabulary error: </strong><br/>
Words that doesn't have pretrained vector (finding a word that we never saw in the training data).
Solution:
	We will replace those out of vocabulary tokens with "UNK" (for unknown) AND train the model on a bigger data corpus.

<hr/>

Data files needed to run the code: <br/>
Training corpus file: 'snli_1.0_train.jsonl'
Testing corpus file: 'snli_1.0_test.jsonl' – 
https://nlp.stanford.edu/projects/snli/

Pre-Trained word embeddings file: 'glove.6B.300d.txt' (6B tokens, 300d vectors) - https://www.kaggle.com/thanakomsn/glove6b300dtxt

Instructions on how to run the code:
<br/>
Save the files in the project directory
Train & Test the model (&save model): run train_and_test.py
(Load model) Predict the model: run predict.py

<hr/>


Although the results from the testing were satisfying, there are some drawbacks in my approach:

#### Drawbacks

<strong> - This approach cannot distinguish between homophones </strong> (dog BARK vs. tree BARK): <br/>
The word bark will be represented using a single token, because they have the same spelling even though the meanings are distinct.

<strong> - Memory intensive: </strong><br/>
Another big drawback is that word embeddings (and word vectors) could be memory intensive.
We have one row and one column for every individual word in our vocabulary.
As my vocabulary size increases, so does the amount of space needed to train the embeddings.

<strong> - The embeddings are very depended on the corpus that was used to train them.</strong><br/>
	Any underline biases that are present in the training data corpus are going to influence the model.
	The closer that we can approximate the text we are going to encounter in the training data - the better.
