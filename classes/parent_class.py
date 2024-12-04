from abc import ABC, abstractmethod
from gliner import GLiNER

class AbstractEvaluator(ABC):

    def __init__(self, model_id, device='cuda:0'):

        self.model = GLiNER.from_pretrained(model_id).to(device)
        self.device = device

    @abstractmethod
    def prepare_dataset(self, dataset, *args, **kwargs):

        pass

    @abstractmethod
    def process_predictions(self, predictions):

        pass

    @abstractmethod
    def __call__(self, texts, labels, *args, **kwargs):

        pass
    
    @abstractmethod
    def compute_f_score(self, predicts, true_labels):

        pass
    
    @abstractmethod
    def evaluate(self, dataset_id, labels=None, *args, **kwargs):

        pass
