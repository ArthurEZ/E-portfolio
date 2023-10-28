import os
os.environ['TZ'] = 'Asia/Bangkok'
import pythainlp as pt
from pythainlp.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
import random
import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random


# Load intents from a JSON file
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            wrds = pt.word_tokenize(pattern, engine="newmm")
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for doc in docs_x:
        bag = []

        for w in words:
            if w in doc:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[docs_x.index(doc)])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Build the neural network using Keras
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the model if it exists, otherwise train it
try:
    model.load('model_keras.h5')
except:
    model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
    model.save('model_keras.h5')

# Define a function to generate a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    correct_words = pt.correct(s)
    s_words = pt.word_tokenize(correct_words, engine="newmm")

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# Chat function
def chat():
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Preprocess input and ensure it has the correct shape
        input_data = bag_of_words(inp, words)
        input_data = input_data.reshape(1, len(input_data))  # Reshape to (1, 73)

        results = model.predict(input_data)[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.7:
            for tg in intents["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))
        else:
            print("ขอโทษนะครับ ผมไม่เข้าใจในคำถาม กรุณาลองใหม่อีกครั้ง")
        print(results_index)

chat()