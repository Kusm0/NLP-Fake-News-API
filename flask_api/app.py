import os
import pandas as pd
from flask import Flask, request, jsonify
from fake_news_model import BERTClassifier, load_model, predict_sentiment, load_data_test, train_and_save
from torch import nn
import torch
from transformers import BertTokenizer, BertModel

app = Flask(__name__)

bert_model_name = 'bert-base-uncased'
num_classes = 2


saved_models_path = "saved_models"
base_model_name = "bert_classifier.pth"
load_path = saved_models_path + "/" + base_model_name


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)

load_model(load_path, model)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    author = data.get('author', '')
    title = data.get('title', '')
    text = data.get('text', '')

    combined_text = load_data_test(author, title, text)
    prediction = predict_sentiment(combined_text, model, tokenizer, device)
    result = 'potentially fake (`1`)' if prediction == 1 else 'reliable (`0`)'

    return jsonify({
        'prediction': result
    })


@app.route('/upload_model_file', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({"error": "No model file provided"}), 400

    file = request.files['model']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    new_file_path = saved_models_path+'/'+file.filename

    file.save(new_file_path)

    error_message = load_model(new_file_path, model)

    if error_message:
        return jsonify({"error": error_message}), 500

    return jsonify({"message": "Model uploaded and loaded successfully"}), 200


@app.route('/upload_model_path', methods=['POST'])
def load_model_path():
    data = request.get_json()
    model_path = data.get('model_path', '')

    if not model_path:
        return jsonify({"error": "No model path provided"}), 400

    error_message = load_model(model_path, model)
    if error_message:
        return jsonify({"error": error_message}), 500

    return jsonify({"message": "Model loaded successfully from path"}), 200


@app.route('/train_and_save_model', methods=['POST'])
def train_and_save_model():
    if 'train_data' not in request.files:
        return jsonify({"error": "No training data provided"}), 400

    train_file = request.files['train_data']
    epoch = int(request.form.get('epoch', 1))

    if train_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if epoch <= 0:
        return jsonify({"error": "Epoch should be > 0"}), 400

    score, model_path = train_and_save(train_file, epoch, saved_models_path)

    # # Тут же деплой
    # error_message = load_model(model_path)
    # if error_message:
    #     return jsonify({"error": error_message}), 500

    return jsonify({"score": score, "trained_model_path": model_path}), 200


@app.route('/get_models', methods=['GET'])
def get_models():

    files = []

    for dirpath, dirnames, filenames in os.walk(saved_models_path):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))

    return jsonify({"models": files}), 200


if __name__ == '__main__':
    app.run(debug=True)
