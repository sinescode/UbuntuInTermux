import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import PyPDF2
import re
import json
import torch
import requests
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Previous logging configuration remains the same
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Enhanced configuration for text processing parameters."""
    min_words_per_chunk: int = 50
    max_sequence_length: int = 1024
    overlap_size: int = 100
    remove_special_chars: bool = False
    lowercase: bool = False
    start_page: int = 5
    test_size: float = 0.2
    n_clusters: int = 5  # For content clustering

class TextAnalyzer:
    """Analyze and evaluate text quality using scikit-learn."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english'
        )
        self.kmeans = KMeans(
            n_clusters=config.n_clusters,
            random_state=42
        )
    
    def analyze_content_distribution(self, texts: List[str]) -> Dict:
        """Analyze content distribution using TF-IDF and clustering."""
        vectors = self.vectorizer.fit_transform(texts)
        clusters = self.kmeans.fit_predict(vectors)
        
        # Get most representative terms per cluster
        cluster_terms = {}
        for i in range(self.config.n_clusters):
            center_idx = clusters == i
            if np.any(center_idx):
                centroid = vectors[center_idx].mean(axis=0)
                terms_scores = [(term, score) for term, score in 
                              zip(self.vectorizer.get_feature_names_out(),
                                  centroid.toarray()[0])]
                top_terms = sorted(terms_scores, key=lambda x: x[1], reverse=True)[:10]
                cluster_terms[f"cluster_{i}"] = top_terms
        
        return {
            "cluster_sizes": np.bincount(clusters).tolist(),
            "representative_terms": cluster_terms
        }

class DjangoDocDatasetProcessor:
    """Enhanced processor with better data splitting and analysis."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.text_analyzer = TextAnalyzer(config)
    
    # Previous methods remain the same until create_dataset
    
    def create_dataset(self, pdf_path: str) -> Tuple[Dataset, Dataset]:
        """
        Convert PDF text into train and test datasets with analysis.
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        pdf_path = Path(pdf_path)
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.preprocess_text(text)
        
        # Analyze content distribution
        texts = [chunk["text"] for chunk in chunks]
        content_analysis = self.text_analyzer.analyze_content_distribution(texts)
        logger.info(f"Content distribution analysis: {json.dumps(content_analysis, indent=2)}")
        
        # Split into train and test sets
        train_chunks, test_chunks = train_test_split(
            chunks,
            test_size=self.config.test_size,
            random_state=42
        )
        
        train_dataset = Dataset.from_list(train_chunks)
        test_dataset = Dataset.from_list(test_chunks)
        
        # Save processed datasets
        output_dir = pdf_path.parent
        train_dataset.save_to_disk(str(output_dir / "train_dataset"))
        test_dataset.save_to_disk(str(output_dir / "test_dataset"))
        
        return train_dataset, test_dataset

class ModelTrainer:
    """Enhanced trainer with improved evaluation metrics."""
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        device: Optional[str] = None
    ):
        # Previous initialization remains the same
        super().__init__()
        self.metrics_history = []
    
    def evaluate_model(self, dataset: Dataset) -> Dict:
        """
        Evaluate model performance using various metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        references = []
        
        eval_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Get predictions
                logits = outputs.logits
                pred = torch.argmax(logits, dim=-1)
                predictions.extend(pred.cpu().numpy())
                references.extend(batch["labels"].cpu().numpy())
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            references,
            predictions,
            average='weighted'
        )
        
        metrics = {
            "loss": total_loss / len(eval_dataloader),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        self.metrics_history.append(metrics)
        return metrics

    def train(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        epochs: int = 5,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        warmup_steps: int = 200
    ):
        """Enhanced training with validation and metrics tracking."""
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=["tensorboard"]
        )
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {
                "accuracy": (predictions == labels).mean(),
                **precision_recall_fscore_support(
                    labels,
                    predictions,
                    average='weighted',
                    zero_division=0
                )
            }
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train and evaluate
        trainer.train()
        final_metrics = trainer.evaluate()
        
        # Save metrics history
        metrics_path = self.output_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                "training_history": self.metrics_history,
                "final_metrics": final_metrics
            }, f, indent=2)
        
        # Save final model and tokenizer
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))

def main():
    config = ProcessingConfig(
        min_words_per_chunk=50,
        max_sequence_length=1024,
        overlap_size=100,
        remove_special_chars=False,
        lowercase=False,
        start_page=5,
        test_size=0.2,
        n_clusters=5
    )
    
    pdf_url = "https://media.readthedocs.org/pdf/django/5.1.x/django.pdf"
    pdf_path = "django.pdf"
    
    try:
        if not Path(pdf_path).exists():
            download_pdf(pdf_url, pdf_path)
        
        processor = DjangoDocDatasetProcessor(config)
        train_dataset, test_dataset = processor.create_dataset(pdf_path)
        
        trainer = ModelTrainer(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            output_dir="./django_trained_model"
        )
        
        trainer.setup_model_and_tokenizer()
        trainer.train(
            train_dataset,
            test_dataset,
            epochs=5,
            batch_size=2,
            learning_rate=1e-5,
            warmup_steps=200
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
