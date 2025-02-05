import logging
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
import re
import json
import torch
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for text processing parameters."""
    min_words_per_chunk: int = 5
    max_sequence_length: int = 512
    overlap_size: int = 50
    remove_special_chars: bool = True
    lowercase: bool = False

class DjangoDocDatasetProcessor:
    """Process Django documentation PDFs for language model fine-tuning."""
    
    def __init__(self, config: ProcessingConfig = ProcessingConfig()):
        self.config = config
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file with error handling and logging.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            PyPDF2.PdfReadError: If PDF is corrupted or unreadable
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            text = ""
            with pdf_path.open('rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"Processing {total_pages} pages from {pdf_path.name}")
                
                for i, page in enumerate(tqdm(pdf_reader.pages, desc="Extracting text")):
                    text += page.extract_text() + "\n\n"
                    
            logger.info(f"Successfully extracted {len(text)} characters")
            return text
            
        except PyPDF2.PdfReadError as e:
            logger.error(f"Failed to read PDF: {e}")
            raise

    def preprocess_text(self, text: str) -> List[Dict[str, str]]:
        """
        Preprocess text with improved chunking and cleaning.
        
        Args:
            text: Input text to process
            
        Returns:
            List of dictionaries containing processed text chunks
        """
        if self.config.lowercase:
            text = text.lower()
            
        if self.config.remove_special_chars:
            text = re.sub(r'[^\w\s.]-', '', text)
            
        # Split into paragraphs while preserving important whitespace
        paragraphs = [p.strip() for p in re.split(r'\n\n+', text)]
        
        # Create overlapping chunks for better context
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            words = para.split()
            if len(words) < self.config.min_words_per_chunk:
                continue
                
            if len(current_chunk.split()) + len(words) > self.config.max_sequence_length:
                if current_chunk:
                    chunks.append({"text": current_chunk.strip()})
                current_chunk = " ".join(words[-self.config.overlap_size:])
            else:
                current_chunk += " " + para
                
        if current_chunk:
            chunks.append({"text": current_chunk.strip()})
            
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks

    def create_dataset(self, pdf_path: str) -> Dataset:
        """
        Convert PDF text into Hugging Face Dataset with validation.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Hugging Face Dataset object
        """
        pdf_path = Path(pdf_path)
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.preprocess_text(text)
        
        # Validate chunks
        valid_chunks = [
            chunk for chunk in chunks 
            if len(chunk["text"].split()) >= self.config.min_words_per_chunk
        ]
        
        dataset = Dataset.from_list(valid_chunks)
        
        # Save processed dataset
        output_path = pdf_path.with_suffix('.json')
        dataset.save_to_disk(str(output_path))
        logger.info(f"Saved processed dataset to {output_path}")
        
        return dataset

class ModelTrainer:
    """Handles model training with improved configuration and monitoring."""
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with proper configuration."""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='right',
                truncation_side='right'
            )
            
            # Add padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training with dynamic batching."""
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )

        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset

    def train(
        self,
        dataset: Dataset,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100
    ):
        """
        Train the model with improved monitoring and checkpointing.
        
        Args:
            dataset: Prepared dataset for training
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            warmup_steps: Number of warmup steps for learning rate scheduler
        """
        tokenized_dataset = self.prepare_dataset(dataset)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=self.device == 'cuda',
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            report_to=["tensorboard"],
            remove_unused_columns=False
        )
        
        # Initialize trainer with data collator
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train with error handling
        try:
            logger.info("Starting training...")
            trainer.train()
            
            # Save final model and tokenizer
            trainer.save_model(str(self.output_dir / "final_model"))
            self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def main():
    # Configuration
    config = ProcessingConfig(
        min_words_per_chunk=10,
        max_sequence_length=512,
        overlap_size=50
    )
    
    # Initialize processor and create dataset
    try:
        processor = DjangoDocDatasetProcessor(config)
        dataset = processor.create_dataset('django_docs.pdf')
        
        # Initialize trainer
        trainer = ModelTrainer(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            output_diimport logging
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
import re
import json
import torch
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for text processing parameters."""
    min_words_per_chunk: int = 5
    max_sequence_length: int = 512
    overlap_size: int = 50
    remove_special_chars: bool = True
    lowercase: bool = False

class DjangoDocDatasetProcessor:
    """Process Django documentation PDFs for language model fine-tuning."""
    
    def __init__(self, config: ProcessingConfig = ProcessingConfig()):
        self.config = config
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file with error handling and logging.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            PyPDF2.PdfReadError: If PDF is corrupted or unreadable
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            text = ""
            with pdf_path.open('rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"Processing {total_pages} pages from {pdf_path.name}")
                
                for i, page in enumerate(tqdm(pdf_reader.pages, desc="Extracting text")):
                    text += page.extract_text() + "\n\n"
                    
            logger.info(f"Successfully extracted {len(text)} characters")
            return text
            
        except PyPDF2.PdfReadError as e:
            logger.error(f"Failed to read PDF: {e}")
            raise

    def preprocess_text(self, text: str) -> List[Dict[str, str]]:
        """
        Preprocess text with improved chunking and cleaning.
        
        Args:
            text: Input text to process
            
        Returns:
            List of dictionaries containing processed text chunks
        """
        if self.config.lowercase:
            text = text.lower()
            
        if self.config.remove_special_chars:
            text = re.sub(r'[^\w\s.]-', '', text)
            
        # Split into paragraphs while preserving important whitespace
        paragraphs = [p.strip() for p in re.split(r'\n\n+', text)]
        
        # Create overlapping chunks for better context
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            words = para.split()
            if len(words) < self.config.min_words_per_chunk:
                continue
                
            if len(current_chunk.split()) + len(words) > self.config.max_sequence_length:
                if current_chunk:
                    chunks.append({"text": current_chunk.strip()})
                current_chunk = " ".join(words[-self.config.overlap_size:])
            else:
                current_chunk += " " + para
                
        if current_chunk:
            chunks.append({"text": current_chunk.strip()})
            
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks

    def create_dataset(self, pdf_path: str) -> Dataset:
        """
        Convert PDF text into Hugging Face Dataset with validation.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Hugging Face Dataset object
        """
        pdf_path = Path(pdf_path)
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.preprocess_text(text)
        
        # Validate chunks
        valid_chunks = [
            chunk for chunk in chunks 
            if len(chunk["text"].split()) >= self.config.min_words_per_chunk
        ]
        
        dataset = Dataset.from_list(valid_chunks)
        
        # Save processed dataset
        output_path = pdf_path.with_suffix('.json')
        dataset.save_to_disk(str(output_path))
        logger.info(f"Saved processed dataset to {output_path}")
        
        return dataset

class ModelTrainer:
    """Handles model training with improved configuration and monitoring."""
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with proper configuration."""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='right',
                truncation_side='right'
            )
            
            # Add padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training with dynamic batching."""
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )

        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset

    def train(
        self,
        dataset: Dataset,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100
    ):
        """
        Train the model with improved monitoring and checkpointing.
        
        Args:
            dataset: Prepared dataset for training
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            warmup_steps: Number of warmup steps for learning rate scheduler
        """
        tokenized_dataset = self.prepare_dataset(dataset)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=self.device == 'cuda',
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            report_to=["tensorboard"],
            remove_unused_columns=False
        )
        
        # Initialize trainer with data collator
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train with error handling
        try:
            logger.info("Starting training...")
            trainer.train()
            
            # Save final model and tokenizer
            trainer.save_model(str(self.output_dir / "final_model"))
            self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def main():
    # Configuration
    config = ProcessingConfig(
        min_words_per_chunk=10,
        max_sequence_length=512,
        overlap_size=50
    )
    
    # Initialize processor and create dataset
    try:
        processor = DjangoDocDatasetProcessor(config)
        dataset = processor.create_dataset('django_docs.pdf')
        
        # Initialize trainer
        trainer = ModelTrainer(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            output_dir="./trained_model"
        )
        
        # Setup and train
        trainer.setup_model_and_tokenizer()
        trainer.train(dataset)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()r="./trained_model"
        )
        
        # Setup and train
        trainer.setup_model_and_tokenizer()
        trainer.train(dataset)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
