import torch
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from text_classification_dataset import TextClassificationDataset
import os


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


def load_model(path, model):
    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully. All keys match.")
        return None  # Немає помилки
    except Exception as e:
        error_message = f"Error loading model: {e}"
        print(error_message)
        return error_message


def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return 0 if preds.item() == 0 else 1


def load_data_test(author, title, text):
    combined_text = author + " " + title + " " + text
    return combined_text


############ Train Functions ###########

def load_data(data_file):
    df = pd.read_csv(data_file)
    df = df.fillna('')
    df['combined_text'] = df['author'] + df['title'] + df['text']
    texts = df['combined_text'].tolist()
    labels = df['label'].tolist()
    return texts, labels


def train(model, data_loader, optimizer, scheduler, device, save_path, num_epoch):
    model.train()
  #Список для зберігання значень лоссу
    for batch_idx, batch in enumerate(data_loader):
        optimizer.zero_grad()
    
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
    
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
        # Assuming the outputs are logits directly
        logits = outputs
    
        # Compute loss
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Збереження моделі після кожної ітерації
    model_save_path = os.path.join(save_path, f"model_epoch_{num_epoch + 1}.pt")
    torch.save(model.state_dict(), model_save_path)
    return model_save_path


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


def train_and_save(train_data, num_epochs, path):
    data_file = train_data
    texts, labels = load_data(data_file)

    bert_model_name = 'bert-base-uncased'
    num_classes = 2
    max_length = 128
    batch_size = 16
    learning_rate = 2e-5

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(bert_model_name, num_classes).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    accuracy_history = [0.99]
    path_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        path_of_model = train(model, train_dataloader, optimizer, scheduler, device, path, epoch)
        accuracy, report = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)
        accuracy_history.append(accuracy)
        path_history.append(path_of_model)

    score = accuracy_history[-1]

    return score, path_history
