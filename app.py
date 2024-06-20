# -*- coding: utf-8 -*-
import json, torch, random, unicodedata, nltk, torch.nn as nn, numpy as np

from flask import Flask, render_template, request, jsonify, Response
from subprocess import Popen, PIPE
from nltk.stem.porter import PorterStemmer

class Red_Neuronal(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Red_Neuronal, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        
        # Función de activación de capa
        self.relu = nn.ReLU()

    def forward(self, data):
        out = self.layer1(data)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out

# Tokenizar
def tokenizer(sentence):
    return nltk.word_tokenize(sentence)

# Stemming
def stemming(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())

def bolsa_de_palabras(tokenized_sentence, words):
    tokenized_sentence = [stemming(i) for i in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32) # type: ignore
    for i, word in enumerate(words):
        if word in tokenized_sentence:
            bag[i] = 1.0

    return bag

# Abrir JSON
with open('./intents/intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

data = torch.load("./data/dictionary.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
num_classes = data["num_classes"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Red_Neuronal(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(model_state)
model.eval()

def normalizar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar acentos
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return texto

def asistente(input_text):
    input_text = normalizar_texto(input_text)
    sentence = tokenizer(input_text)
    bag = bolsa_de_palabras(sentence, all_words)
    bag = bag.reshape(1, bag.shape[0])
    bag = torch.from_numpy(bag).to(device)

    output = model(bag)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    prob = probabilities[0][predicted.item()].item() # type: ignore

    if prob > 0.80:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['response'])
    else:
        return "Para esa duda o inconveniente deberá contactar con nuestro equipo de desarrollo."


class Assistant():

    def chat(self,text):
        chat = asistente(text)
        return chat

# API Flask
app = Flask(__name__) 
selena = Assistant()

# Método para responder a las consultas de los usuarios
@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '') # type: ignore
    bot_response = selena.chat(user_message)
    return jsonify({'response': bot_response})

# Método para entrenar el modelo
@app.route('/api/train', methods=['POST'])
def train():
    # Recibir datos de entrenamiento
    training_data = request.json

    # Guardar los datos en el archivo intents.json
    with open('./intents/intents.json', 'w', encoding='utf-8') as outfile:
        json.dump(training_data, outfile, indent=4, ensure_ascii=False)

    # Ejecutar el script de entrenamiento
    process = Popen(['python', './data/train.py'], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    while process.poll() is None:
        line = process.stdout.readline() # type: ignore
        if line:
            yield f"data:{line.strip()}\n\n"
    yield f"data:Entrenamiento completado\n\n"

if __name__ == '__main__':
    app.run(debug=True)