from gliner import GLiNER
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
from datetime import datetime


class MultiTaskProcessor:

    def __init__(self, model_name="knowledgator/gliner-multitask-large-v0.5", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiNER.from_pretrained(model_name).to(self.device)
        self.tasks = {
            "ner": self._process_ner,
            "relation_extraction": self._process_relation_extraction,
            "summarize": self._process_summarize,
            "open_extraction": self._process_open_extraction,
            "question_answer": self._process_question_answer,
            "sentiment_analysis": self._sentiment_analysis,
        }

    def _save_results(self, results, task_name):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{task_name}_results_{timestamp}.json"

        with open(file_name, "w") as file:
            json.dump(results, file, indent=2)

    def _process_ner(self, dataset, threshold=0.5):

        labels = ["Person", "Country", "Location", "City", "Event", "Organization", 'State']
        results = {}

        with tqdm(total=len(dataset), desc="Processing NER") as pbar:
            for idx, entry in enumerate(dataset):
                text = " ".join(entry["tokenized_text"])
                entities = self.model.predict_entities(text, labels, flat_ner=False, threshold=threshold)
                results[idx] = entities
                pbar.update(1)

        return results

    def _process_relation_extraction(self, dataset, threshold=0.5):
        results = {}

        with open(dataset, 'r', encoding='utf-8') as f:
            with tqdm(desc="Processing Relation Extraction", unit="entry") as pbar:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        tokens = entry.get("tokens", [])
                        head = entry.get("head", {})
                        head_name = head.get("text", None)
                        relations = entry.get("names", [])

                        if not head_name or not tokens or not relations:
                            pbar.update(1)
                            continue

                        labels = [f"{head_name} <> {relation}" for relation in relations]

                        input_text = f"Context: " + " ".join(tokens)

                        predictions = self.model.predict_entities(
                            input_text, labels=labels, flat_ner=False, threshold=threshold
                        )

                        for prediction in predictions:
                            predicted_tail = prediction.get("text")
                            label = prediction.get("label")
                            if predicted_tail and label:

                                predicted_tail = predicted_tail.lower()

                                result_str = f"{label} <> {predicted_tail}"

                                if result_str not in results:
                                    results[result_str] = []
                                results[result_str].append({
                                    'head_relation': label,
                                    'tail': predicted_tail,
                                    'text': input_text
                                })

                        pbar.update(1)
                        torch.cuda.empty_cache()
                    except json.JSONDecodeError as e:
                        print(f"Error reading JSON: {e}")
                    except Exception as e:
                        print(f"Error: {e}")

        return results




    def _process_summarize(self, dataset, threshold=0.5):
        labels = ['summary']
        prompt = "Summarize the given text, highlighting the most important information:\n"
        results = {}

        with tqdm(total=len(dataset), desc="Processing Summarization") as pbar:
            for idx, entry in enumerate(dataset):
                text = entry["article"]
                input_text = prompt + text
                summaries = self.model.predict_entities(input_text, labels=labels, threshold=threshold)

                summary_text = [summary["text"] for summary in summaries]
                results[idx] = summary_text
                pbar.update(1)

        return results

    def _process_question_answer(self, dataset, threshold=0.5):
        predictions = []

        with tqdm(total=len(dataset), desc="Processing Question Answering") as pbar:
            for entry in dataset:
                question_id = entry["id"]
                question = entry["question"]
                context = entry["context"]

                input_text = question + " " + context

                answers = self.model.predict_entities(input_text, labels=["answer"], threshold=threshold)

                prediction = {
                    "id": question_id,
                    "prediction_text": answers[0]["text"] if answers else "",
                }

                predictions.append(prediction)
                pbar.update(1)

        return predictions



    def _sentiment_analysis(self, dataset, threshold=0.5):

      labels = ["positive", "negative"]
      results = {}
      with tqdm(total=len(dataset), desc="Processing Sentiment Analysis") as pbar:
          for idx, entry in enumerate(dataset):
              text = entry["text"]
              if not text.strip():
                  results[idx] = {"text": text, "sentiment": []}
                  pbar.update(1)
                  continue
              try:
                  sentiment_result = self.model.predict_entities(
                      text, labels=labels, flat_ner=False, threshold=threshold
                  )
                  results[idx] = {
                      "text": text,
                      "sentiment": [
                          {"label": entity["label"], "score": entity["score"]}
                          for entity in sentiment_result
                      ],
                  }
              except Exception as e:
                  results[idx] = {"text": text, "sentiment": []}
              pbar.update(1)
      return results


    def _process_open_extraction(self, dataset, threshold=0.5):
        labels = ["match"]
        results = {}

        with tqdm(total=len(dataset), desc="Processing Positive Aspects") as pbar:
            for idx, entry in enumerate(dataset):

                text = entry["text"]
                prompt = "Find all positive aspects about the product:\n"
                input_text = prompt + text

                matches = self.model.predict_entities(input_text, labels=labels, threshold=threshold)

                positive_aspects = [match["text"] for match in matches]
                results[idx] = positive_aspects
                pbar.update(1)

        return results

    def process(self, task_name, dataset, threshold=0.5):

        if task_name not in self.tasks:
            raise ValueError(f"invalid task name: {task_name}")

        results = self.tasks[task_name](dataset, threshold)
        self._save_results(results, task_name)

        return results


if __name__ == "__main__":
    
    processor = MultiTaskProcessor()

    results_re = processor.process("relation_extraction", "val_wiki-2.json")
    processor._save_results(results_re, "relation_extraction")
