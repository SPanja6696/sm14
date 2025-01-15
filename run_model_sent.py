import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm

# Set paths
MODEL_PATH = "C:/Users/oushn/OneDrive/Documents/programs/projects/Alinsight/Allinsight/sentiment_model/sentiment_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
print("[INFO] Loading the model...")
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.to(device)
print("[INFO] Model loaded successfully.\n")

# Pandas option to display full text
pd.set_option('display.max_colwidth', None)

# User chooses between file analysis or manual input
print("Choose the mode of sentiment analysis:")
print("1. Analyze CSV file")
print("2. Test custom sentences")
choice = input("Enter 1 or 2: ").strip()
# Define the predict_sentiment function
def predict_sentiment(statement):
    encoding = tokenizer(statement, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    sentiment = torch.argmax(probs).item()
    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    return sentiment_label, probs[0].cpu().numpy()
if choice == "1":
    # Display available CSV files
    print("\n[INFO] Please place the CSV file in the current directory.")
    csv_files = [f for f in os.listdir() if f.endswith(".csv")]
    if not csv_files:
        print("[ERROR] No CSV files found in the directory.")
        exit()
    print("Available CSV files:")
    for i, file in enumerate(csv_files, start=1):
        print(f"{i}. {file}")

    # User selects a file
    try:
        file_index = int(input("\nEnter the number corresponding to the CSV file you want to analyze: "))
        selected_file = csv_files[file_index - 1]
        print(f"[INFO] You selected: {selected_file}")
    except (ValueError, IndexError):
        print("[ERROR] Invalid selection. Exiting.")
        exit()

    # Load selected file
    df = pd.read_csv(selected_file)
    text_column = "Text"  # Update this if your text column has a different name
    if text_column not in df.columns:
        print(f"[ERROR] Column '{text_column}' not found in the CSV file.")
        exit()

    statements = df[text_column].tolist()
    
    # Define the predict_sentiment function
    def predict_sentiment(statement):
        encoding = tokenizer(statement, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
        with torch.no_grad():
            outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        sentiment = torch.argmax(probs).item()
        sentiment_label = "Positive" if sentiment == 1 else "Negative"
        return sentiment_label, probs[0].cpu().numpy()

    # Perform sentiment analysis
    print("\n[INFO] Performing sentiment analysis on the first few statements...\n")
    results = []
    positive_count = 0
    negative_count = 0

    for statement in tqdm(statements, desc="Analyzing Sentiments"):
        sentiment_label, probabilities = predict_sentiment(statement)
        results.append({"statement": statement, "sentiment": sentiment_label, "probabilities": probabilities.tolist()})
        if sentiment_label == "Positive":
            positive_count += 1
        else:
            negative_count += 1

    # Assign sentiments to the DataFrame
    df["Sentiment"] = pd.Series([result["sentiment"] for result in results])

    # Display first few rows with sentiments
    print("\nStatement-wise Sentiment Analysis (First 5 rows):")
    print(df[[text_column, "Sentiment"]].head())

    # Visualization
    print("[INFO] Generating visualizations...\n")
    sentiment_counts = df["Sentiment"].value_counts()

    # Pie Chart
    plt.figure(figsize=(8, 6))
    colors = ["skyblue", "salmon"]
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", colors=colors, startangle=140)
    plt.title("Sentiment Distribution (Pie Chart)")
    plt.show()

    # Bar Chart
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
    plt.title("Sentiment Counts (Bar Chart)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

elif choice == "2":
    print("\n[INFO] Custom Sentence Testing Mode")
    print("Type 'exit' to stop testing.\n")
    while True:
        user_input = input("Enter a sentence for sentiment analysis: ").strip()
        if user_input.lower() == "exit":
            print("[INFO] Exiting custom sentence testing.")
            break
        sentiment_label, probabilities = predict_sentiment(user_input)
        print(f"Sentence: {user_input}")
        print(f"Predicted Sentiment: {sentiment_label}")
        print(f"Confidence: Positive: {probabilities[1]:.2f}, Negative: {probabilities[0]:.2f}\n")

else:
    print("[ERROR] Invalid choice. Exiting.")
    exit()
