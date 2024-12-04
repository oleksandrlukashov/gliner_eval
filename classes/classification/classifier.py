import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parent_class import AbstractEvaluator
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score
from gliner import GLiNER

class GLiNERclassEvaluator(AbstractEvaluator):
    """
    A class to evaluate the GLiNER model for classification tasks using F1 scores.

    Attributes:
        device (str): Device to run the model on, e.g., 'cuda:0' or 'cpu'.
        model (GLiNER): Loaded GLiNER model instance.
        prompt (str): Template prompt for text classification.

    Methods:
        compute_f_score(predicts, true_labels):
            Computes micro, macro, and weighted F1 scores.
        prepare_dataset(dataset, classes=None, text_column='text', label_column='label', split=None, max_examples=-1):
            Prepares texts and true labels from the given dataset.
        process_predictions(predictions):
            Processes model predictions to extract the most likely labels.
        prepare_texts(texts, labels):
            Creates classification prompts for each input text.
        __call__(texts, labels, threshold=0.5):
            Runs the model on the given texts and returns predicted labels.
        evaluate(dataset_id, labels=None, threshold=0.5, max_examples=-1):
            Evaluates the model on a dataset and computes F1 scores.
    """

    def __init__(self, model_id, device='cuda:0'):
        """
        Initializes the GLiNERclassEvaluator with a pretrained GLiNER model.

        Args:
            model_id (str): Identifier for the pretrained GLiNER model.
            device (str): Device to run the model on, default is 'cuda:0'.
        """
        self.device = device
        self.model = GLiNER.from_pretrained(model_id).to(device)
        self.prompt = """Classify the given text having the following classes: {}"""

    def compute_f_score(self, predicts, true_labels):
        """
        Computes the micro, macro, and weighted F1 scores.

        Args:
            predicts (list): List of predicted labels.
            true_labels (list): List of true labels.

        Returns:
            dict: Dictionary with micro, macro, and weighted F1 scores.
        """
        micro = f1_score(true_labels, predicts, average="micro")
        macro = f1_score(true_labels, predicts, average="macro")
        weighted = f1_score(true_labels, predicts, average="weighted")
        return {"micro": micro, "macro": macro, "weighted": weighted}

    def prepare_dataset(self, dataset, classes=None, text_column='text', label_column="label", split=None, max_examples=-1):
        """
        Prepares the dataset by extracting texts and true labels.

        Args:
            dataset (Dataset or dict): The dataset to prepare.
            classes (list, optional): List of class labels. Defaults to None.
            text_column (str): Name of the text column. Defaults to 'text'.
            label_column (str): Name of the label column. Defaults to 'label'.
            split (str, optional): Delimiter for splitting class names. Defaults to None.
            max_examples (int): Maximum number of examples to use. Defaults to -1 (use all).

        Returns:
            tuple: Texts, classes, and true labels.
        """
        if 'test' in dataset:
            test_dataset = dataset['test']
        elif isinstance(dataset, Dataset):
            test_dataset = dataset
        else:
            test_dataset = dataset['train']
        
        if classes is None:
            classes = test_dataset.features[label_column].names
            if split is not None:
                classes = [' '.join(class_.split(split)) for class_ in classes]

        texts = test_dataset[text_column]
        true_labels = test_dataset[label_column]

        if isinstance(test_dataset[label_column][0], int):
            true_labels = [classes[label] for label in true_labels]

        if max_examples > 0:
            texts = texts[:max_examples]
            true_labels = true_labels[:max_examples]

        return texts, classes, true_labels

    def process_predictions(self, predictions):
        """
        Processes predictions to extract the highest-scoring label.

        Args:
            predictions (list): List of predictions with scores.

        Returns:
            list: List of predicted labels.
        """
        predicted_labels = []
        for prediction in predictions:
            prediction = sorted(prediction, key=lambda entity: entity["score"], reverse=True)
            label = prediction[0]['text'] if prediction else 'other'
            predicted_labels.append(label)
        return predicted_labels

    def prepare_texts(self, texts, labels):
        """
        Prepares prompts for classification by appending labels to texts.

        Args:
            texts (list): List of input texts.
            labels (list): List of classification labels.

        Returns:
            list: List of formatted prompts.
        """
        prompts = []
        labels_ = ', '.join(labels)
        for text in texts:
            prompt = self.prompt.format(labels_) + ' \n ' + text
            prompts.append(prompt)
        return prompts

    def __call__(self, texts, labels, threshold=0.5):
        """
        Runs the model on the provided texts and returns predicted labels.

        Args:
            texts (list): List of input texts.
            labels (list): List of class labels.
            threshold (float): Threshold for prediction confidence. Defaults to 0.5.

        Returns:
            list: List of predicted labels.
        """
        prompts = self.prepare_texts(texts, labels)
        predictions = self.model.run(prompts, ['match'], threshold=threshold)
        return self.process_predictions(predictions)

    def evaluate(self, dataset_id, labels=None, threshold=0.5, max_examples=-1):
        """
        Evaluates the model on a specified dataset.

        Args:
            dataset_id (str): Identifier for the dataset to evaluate.
            labels (list, optional): List of labels. Defaults to None.
            threshold (float): Threshold for prediction confidence. Defaults to 0.5.
            max_examples (int): Maximum number of examples to use. Defaults to -1 (use all).

        Returns:
            dict: Evaluation results containing F1 scores.
        """
        dataset = load_dataset(dataset_id)
        test_texts, classes, true_labels = self.prepare_dataset(dataset, labels, max_examples=max_examples)
        print('Test dataset length:', len(test_texts))
        
        predictions = self.__call__(test_texts, classes, threshold=threshold)
        return self.compute_f_score(predictions, true_labels)
