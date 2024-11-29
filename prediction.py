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
        labels = ["<entity1> <> <relation> <> <entity2>"]
        results = {}

        with tqdm(total=len(dataset), desc="Processing Relation Extraction") as pbar:
            for idx, entry in enumerate(dataset):
                text = " ".join(entry["tokenized_text"])
                relations = self.model.predict_entities(text, labels, flat_ner=False, threshold=threshold)
                results[idx] = relations
                pbar.update(1)

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
        results = {}

        with tqdm(total=len(dataset), desc="Processing Question Answering") as pbar:
            for idx, entry in enumerate(dataset):
                question = entry["question"]
                context = entry["context"]
                input_text = question + " " + context

                answers = self.model.predict_entities(input_text, labels=["answer"], threshold=threshold)

                results[idx] = []
                for answer in answers:
                    results[idx].append({
                        "answer": answer["text"],
                        "score": answer["score"]
                    })
                pbar.update(1)

        return results

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
            raise ValueError(f"Invalid task name: {task_name}")

        results = self.tasks[task_name](dataset, threshold)
        self._save_results(results, task_name)

        return results


if __name__ == "__main__":

    with open("eval/ner/unie_synthetic.json", "r") as file:
        dataset_ner_re = json.load(file)

    dataset_sum_keywords = load_dataset('cnn_dailymail', '3.0.0')['validation']
    dataset_qa = load_dataset("rajpurkar/squad_v2")['validation']
    dataset_sment_open = load_dataset("stanfordnlp/imdb")['test']

    processor = MultiTaskProcessor()


    results_re = processor.process("relation_extraction", dataset_ner_re)
    processor._save_results(results_re, "relation_extraction")

    results_qa = processor.process('question_answer', dataset_qa)
    processor._save_results(results_qa, 'question_answer')

    results_sum = processor.process("summarize", dataset_sum_keywords)
    processor._save_results(results_sum, 'summarize')

    results_sment = processor.process('sentiment_analysis', dataset_sment_open)
    processor._save_results(results_sment, 'sentiment_analysis')

    results_open = processor.process('open_extraction', dataset_sment_open)
    processor._save_results(results_open, 'open_extraction')
