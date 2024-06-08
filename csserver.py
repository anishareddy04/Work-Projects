import socket
import threading
import pickle
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import json
import random
import argparse
import logging
import time
import os
import ssl

stemmer = PorterStemmer()

# Set up logging
def setup_logging(port):
    logging.basicConfig(filename=f'server_{port}.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Define the Neural Network model
class NeuralNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

# Load the model and metadata
def load_model(file_path):
    logging.info(f"Attempting to load model from {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"Model file not found: {file_path}")
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetModel(data['input_size'], data['hidden_size'], data['output_size']).to(device)
    model.load_state_dict(data['model_state'])
    return model, data, device

# Tokenization and stemming utilities
def tokenize(sentence):
    nltk.download('punkt', quiet=True)
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Chatbot response generator
def chatbot_response(model, data, input_text, device):
    model.eval()
    sentence = tokenize(input_text)
    X = bag_of_words(sentence, data['all_words'])
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = data['tags'][predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    with open(r'/home/akhilesh/Desktop/DS+Comp_sec/proj/dataset.json', 'r') as f:
        intents_data = json.load(f)

    intents = intents_data['intents']

    if prob.item() > 0.75:
        for intent in intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

    return "I do not understand..."

def bargain(input_text):
    tokens = nltk.word_tokenize(input_text)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    numbers = {str(i): i for i in range(1, 31)}
    for word, wordType in entities:
        if wordType == 'CD':
            number = int(word)
            if number > 20:
                return "Yes agreed! Now, you can buy the item at this price."
            else:
                return "Sorry, this is too low. Can you raise this price a little bit?"
    return "I do not understand..."

# Simulate a load balancing algorithm
def load_balancing_algorithm():
    # Simulate some computation or decision making for load balancing
    time.sleep(random.uniform(0.05, 0.2))  # Simulate a delay

# Server setup
def handle_client(conn, addr, model, data, device):
    logging.info(f"Connected by {addr}")
    print(f"Connected by {addr}")

    while True:
        try:
            start_time = time.time()  # Start time for measuring response time

            message = conn.recv(1024)
            if not message:
                break
            input_text = message.decode()
            logging.info(f"Received from {addr}: {input_text}")

            # Measure load balancing algorithm time
            lb_start_time = time.time()
            load_balancing_algorithm()
            lb_end_time = time.time()
            lb_time = lb_end_time - lb_start_time
            logging.info(f"Load balancing time: {lb_time:.4f}s")

            # time.sleep(random.uniform(0.1, 1.0))

            if any(char.isdigit() for char in input_text):
                response = bargain(input_text)
            else:
                response = chatbot_response(model, data, input_text, device)
            
            logging.info(f"Responded to {addr}: {response}")
            end_time = time.time()  # End time for measuring response time
            response_time = end_time - start_time
            response_with_time = f"{response} (Running Time: {response_time:.4f}s, Load Balancing Time: {lb_time:.4f}s)"
            conn.sendall(response_with_time.encode())
        except ConnectionResetError:
            logging.error(f"Connection reset by peer: {addr}")
            print(f"Connection reset by peer: {addr}")
            break
        except Exception as e:
            logging.error(f"Error: {e}")
            print(f"Error: {e}")
            break

    conn.close()

def server(port):
    setup_logging(port)
    host = '127.0.0.1'

    # Paths to SSL certificate and key files
    certfile = r'/home/akhilesh/Desktop/DS+Comp_sec/cs/server.crt'
    keyfile = r'/home/akhilesh/Desktop/DS+Comp_sec/cs/server.key'

    # Check if the certificate and key files exist
    if not os.path.exists(certfile) or not os.path.exists(keyfile):
        logging.error(f"SSL certificate or key file not found. Make sure '{certfile}' and '{keyfile}' are in the correct directory.")
        print(f"SSL certificate or key file not found. Make sure '{certfile}' and '{keyfile}' are in the correct directory.")
        return

    # Create an SSL context
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen()
    logging.info(f"Server started. Listening on {host}:{port}")
    print(f"Server started. Listening on {host}:{port}")

    file_path = r'/home/akhilesh/Desktop/DS+Comp_sec/proj/smtg/trained_chatbot.pkl'
    print(f"Using model file path: {file_path}")
    logging.info(f"Using model file path: {file_path}")
    
    if not os.path.exists(file_path):
        logging.error(f"Model file not found at path: {file_path}")
        print(f"Model file not found at path: {file_path}")
        return

    try:
        model, data, device = load_model(file_path)
    except FileNotFoundError as e:
        logging.error(f"Failed to start server: {e}")
        print(f"Failed to start server: {e}")
        return

    while True:
        conn, addr = server_socket.accept()
        ssl_conn = context.wrap_socket(conn, server_side=True)
        thread = threading.Thread(target=handle_client, args=(ssl_conn, addr, model, data, device))
        thread.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chatbot Server')
    parser.add_argument('--port', type=int, default=8443, help='Port to run the server on')
    args = parser.parse_args()
    server(args.port)
