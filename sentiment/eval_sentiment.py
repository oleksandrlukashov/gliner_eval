import json
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_sentiment_analysis(predictions_file, dataset_name="imdb"):

    with open(predictions_file, "r") as file:
        predictions = json.load(file)

    dataset = load_dataset(dataset_name, split="test")

    y_true = []
    y_pred = []

    for idx, data in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
        true_label = "positive" if data["label"] == 1 else "negative"


        if str(idx) not in predictions or not predictions[str(idx)]["sentiment"]:
            y_true.append(true_label)
            y_pred.append("negative" if true_label == "positive" else "positive")
            continue

        label_scores = {"positive": 0.0, "negative": 0.0}
        for item in predictions[str(idx)]["sentiment"]:
            label = item["label"]
            score = item["score"]
            label_scores[label] += score

        final_label = max(label_scores, key=label_scores.get)
        y_true.append(true_label)
        y_pred.append(final_label)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="positive", average="binary")
    recall = recall_score(y_true, y_pred, pos_label="positive", average="binary")
    f1 = f1_score(y_true, y_pred, pos_label="positive", average="binary")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

predictions_path = "sentiment_analysis_results_20241128_222826.json"
metrics = evaluate_sentiment_analysis(predictions_path)
