from typing import Any
import pandas as pd
from tqdm import tqdm

from .whowhat_metrics import SimilarityMetric, DivergencyMetric


class Evaluator():
    def __init__(self,
                 tokenizer: Any=None,
                 text_gen_pipeline: Any=None,
                 gt_data: str=None,
                 test_data_path: str = None,
                 metrics = ("similarity", "divergency"),
                 similarity_model_id: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        assert text_gen_pipeline is not None or gt_data is not None, "Text generation pipeline for evaluation or ground trush data must be defined"

        self.test_data_path = test_data_path
        self.metrics = metrics
        
        if text_gen_pipeline:
            self.gt_data = self._compute_metrics(text_gen_pipeline)
        else:
            self.gt_data = pd.read_csv(gt_data)

        self.similarity = None
        self.divergency = None
        if 'similarity' in self.metrics:
            self.similarity = SimilarityMetric(similarity_model_id)
        if 'divergency' in self.metrics:
            assert tokenizer is not None
            self.divergency = DivergencyMetric(tokenizer)
            
    
    def dump_gt(self, csv_name: str):
        self.gt_data.to_csv(csv_name)
    
    
    def score(self, text_gen_pipeline):
        predictions = self._compute_metrics(text_gen_pipeline, )
        
        all_metrics_per_question = {}
        all_metrics = {}
        
        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(self.gt_data, predictions)
            all_metrics.update(metric_dict)
            all_metrics_per_question.update(metric_per_question)
        
        if self.divergency:
            metric_dict, metric_per_question = self.divergency.evaluate(self.gt_data, predictions)
            all_metrics.update(metric_dict)
            all_metrics_per_question.update(metric_per_question)

        return pd.DataFrame(all_metrics_per_question), pd.DataFrame([all_metrics])
        
    
    def _compute_metrics(self, text_gen_pipeline):
        data = pd.read_csv(self.test_data_path)
        questions = data['questions']
        
        answers = []
        
        for q in tqdm(questions.values, desc="Evaluate pipeline"):
            out = text_gen_pipeline(q)
            out = out[0]['generated_text']
            answers.append(out[len(q):])

        res_data = {'questions': list(questions.values), 'answers': answers}
        df = pd.DataFrame(res_data)
    
        return df
