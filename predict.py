import numpy as np
from helper import PreprocessTextsAndHypotheses, LoadModelAndTokenizer

Entailment = 'entailment'
NonEntailment = 'non-entailment'
Neutral = 'neutral'
Contradiction = 'contradiction'
two_way_class_names = [Entailment, NonEntailment]
three_way_class_names = [Neutral, Entailment, Contradiction]


# old man => superhero


def Predict(text, hypothesis, modelName, classNames):
    model, tokenizer = LoadModelAndTokenizer(modelName)
    x_text, x_hypo = PreprocessTextsAndHypotheses(tokenizer, text, hypothesis)
    probabilities = model.predict([x_text, x_hypo])
    Class = classNames[np.argmax(probabilities[0])]

    return Class


if __name__ == "__main__":
    text = input("Enter Text: ")
    hypothesis = input("Enter Hypothesis: ")
    twoWayClass = Predict(text, hypothesis, '2_way_model', two_way_class_names)
    print("Two Way Prediction: ", twoWayClass)
    threeWayClass = Predict(text, hypothesis, '3_way_model', three_way_class_names)
    print("Three Way Prediction: ", threeWayClass)
