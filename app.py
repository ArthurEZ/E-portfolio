from flask import Flask, request, jsonify, render_template
import pythainlp as pt
import numpy as np
import random
import json
import pickle
from tensorflow.keras.models import load_model
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load chatbot model
model = load_model('model_keras.h5')

# Load intents from a JSON file
with open('/home/ArthurThanagorn/mysite/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Load the words and labels data
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# Function to process input and get a response
def get_response(input_text):
    input_data = bag_of_words(input_text, words)
    input_data = input_data.reshape(1, len(input_data))
    results = model.predict(input_data)[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    if results[results_index] > 0.7:
        for tg in intents["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        return random.choice(responses)
    else:
        return "ขอโทษนะครับ ผมไม่เข้าใจในคำถาม กรุณาลองใหม่อีกครั้ง"

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = pt.word_tokenize(s)
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

@app.route('/')
def home():
    return render_template('index.html')

# Define a route for handling chatbot requests
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('user_message')
    bot_response = get_response(user_message)
    return jsonify({'bot_response': bot_response})


if __name__ == '__main__':
    app.run()