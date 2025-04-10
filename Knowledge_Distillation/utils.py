#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Knowledge Distillation Utilities for LLMs

This module provides utilities for knowledge distillation between two LLM models
loaded from Hugging Face. It focuses on vanilla knowledge distillation with soft targets
and includes memory optimization techniques for efficient training.

The implementation is designed to work with the HPAI-BSC/MMLU-medical-cot-llama31 dataset,
which contains medical questions with chain-of-thought reasoning.
"""

import os
import logging
import gc
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dotenv import load_dotenv
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources for METEOR score
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {e}")


def load_env_variables() -> str:
    """
    Load API keys from .env file.
    
    Returns:
        str: Hugging Face API key
    """
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError(
            "HUGGINGFACE_API_KEY not found in .env file. "
            "Please add it to your .env file."
        )
    return api_key


class MemoryManager:
    """
    Utilities for tracking and optimizing memory usage during training and inference.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the memory manager.
        
        Args:
            verbose (bool): Whether to print memory usage information
        """
        self.verbose = verbose
    
    def print_gpu_utilization(self) -> None:
        """
        Print current GPU memory usage.
        """
        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping GPU memory reporting")
            return
            
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        
        if self.verbose:
            logger.info(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    def estimate_max_batch_size(
        self, 
        model: PreTrainedModel, 
        input_shape: Tuple[int, int], 
        dtype: torch.dtype = torch.float16
    ) -> int:
        """
        Empirically determine maximum batch size for available GPU.
        
        Args:
            model: The model to test
            input_shape: Shape of input tensors (seq_len, hidden_dim)
            dtype: Data type for computation
            
        Returns:
            int: Estimated maximum batch size
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, returning default batch size of 4")
            return 4
            
        # Start with a small batch size
        batch_size = 1
        max_memory = torch.cuda.get_device_properties(0).total_memory
        
        # Create a sample input
        seq_len, hidden_dim = input_shape
        
        try:
            while True:
                # Double the batch size each time
                batch_size *= 2
                
                # Create dummy input
                dummy_input = torch.zeros(
                    (batch_size, seq_len), 
                    dtype=torch.long, 
                    device=model.device
                )
                
                # Try a forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Check memory usage
                if torch.cuda.memory_allocated() > 0.9 * max_memory:
                    # If we're using more than 90% of memory, go back to the previous batch size
                    batch_size //= 2
                    break
                    
                # Safety check
                if batch_size > 128:
                    logger.warning("Stopping batch size estimation at 128")
                    break
                    
        except RuntimeError:
            # Out of memory, go back to the previous batch size
            batch_size //= 2
            torch.cuda.empty_cache()
            
        if self.verbose:
            logger.info(f"Estimated maximum batch size: {batch_size}")
            
        return max(1, batch_size)
    
    def optimize_inference_memory(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Apply inference optimizations to reduce memory usage.
        
        Args:
            model: The model to optimize
            
        Returns:
            The optimized model
        """
        # Enable eval mode
        model.eval()
        
        # Apply torch.compile if available (PyTorch 2.0+)
        if hasattr(torch, 'compile') and callable(getattr(torch, 'compile')):
            try:
                model = torch.compile(model)
                if self.verbose:
                    logger.info("Applied torch.compile to model")
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile: {e}")
        
        return model
    
    def enable_cpu_offloading(
        self, 
        model: PreTrainedModel, 
        device_map: str = "auto"
    ) -> PreTrainedModel:
        """
        Configure CPU offloading for larger models.
        
        Args:
            model: The model to configure
            device_map: Device mapping strategy
            
        Returns:
            The model with CPU offloading configured
        """
        try:
            from accelerate import dispatch_model
            
            # Apply device map
            model = dispatch_model(model, device_map=device_map)
            
            if self.verbose:
                logger.info(f"Enabled CPU offloading with device map: {device_map}")
                
        except ImportError:
            logger.warning(
                "accelerate library not found. "
                "Install with: pip install accelerate"
            )
        except Exception as e:
            logger.warning(f"Failed to enable CPU offloading: {e}")
            
        return model
    
    def cleanup(self) -> None:
        """
        Perform memory cleanup operations.
        """
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Run garbage collection
        gc.collect()
        
        if self.verbose:
            logger.info("Memory cleanup performed")
            self.print_gpu_utilization()


class ModelManager:
    """
    Flexible model loader for teacher and student models with memory optimization.
    """
    
    def __init__(
        self, 
        api_key: str, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        low_cpu_mem_usage: bool = True
    ):
        """
        Initialize the model manager.
        
        Args:
            api_key: Hugging Face API key
            device: Device to load models on ('cuda', 'cpu', etc.)
            low_cpu_mem_usage: Whether to use low CPU memory usage when loading models
        """
        self.api_key = api_key
        self.device = device
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.memory_manager = MemoryManager()
        
        # Set Hugging Face token
        os.environ["HUGGINGFACE_TOKEN"] = api_key
        
        logger.info(f"ModelManager initialized with device: {device}")
    
    def load_model(
        self, 
        model_name: str, 
        is_teacher: bool = False,
        dtype: Optional[torch.dtype] = None,
        use_cpu_offloading: bool = False,
        use_8bit: bool = False,
        use_4bit: bool = False
    ) -> PreTrainedModel:
        """
        Load a model from Hugging Face with memory optimizations.
        
        Args:
            model_name: Name of the model on Hugging Face
            is_teacher: Whether this is the teacher model
            dtype: Data type for model weights
            use_cpu_offloading: Whether to enable CPU offloading
            use_8bit: Whether to load in 8-bit precision
            use_4bit: Whether to load in 4-bit precision
            
        Returns:
            The loaded model
        """
        logger.info(f"Loading model: {model_name} (teacher: {is_teacher})")
        
        # Set default dtype if not provided
        if dtype is None:
            if self.device == 'cuda':
                dtype = torch.float16  # Use FP16 by default on GPU
            else:
                dtype = torch.float32  # Use FP32 by default on CPU
        
        # Configure loading options
        load_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "token": self.api_key,
        }
        
        # Add quantization options if requested
        if use_8bit:
            load_kwargs["load_in_8bit"] = True
        elif use_4bit:
            load_kwargs["load_in_4bit"] = True
        
        try:
            # Load the model
            model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
            
            # Move to device if not using CPU offloading
            if not use_cpu_offloading and not use_8bit and not use_4bit:
                model = model.to(self.device)
            
            # Apply CPU offloading if requested
            if use_cpu_offloading:
                model = self.memory_manager.enable_cpu_offloading(model)
            
            # Optimize for inference if this is the teacher model
            if is_teacher:
                model = self.memory_manager.optimize_inference_memory(model)
                model.eval()  # Ensure teacher is in eval mode
            
            logger.info(f"Successfully loaded model: {model_name}")
            self.memory_manager.print_gpu_utilization()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def load_tokenizer(
        self, 
        model_name: str, 
        use_fast: bool = True
    ) -> PreTrainedTokenizer:
        """
        Load a tokenizer from Hugging Face.
        
        Args:
            model_name: Name of the model on Hugging Face
            use_fast: Whether to use the fast tokenizer implementation
            
        Returns:
            The loaded tokenizer
        """
        logger.info(f"Loading tokenizer for: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=use_fast,
                token=self.api_key
            )
            
            # Ensure the tokenizer has padding token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.eos_token = "</s>"
            
            logger.info(f"Successfully loaded tokenizer for: {model_name}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            raise



def medical_dataset_collate_fn(batch):
    """
    Collate function for MMLUMedicalDataset that handles streaming and pre-processed data.
    """
    if isinstance(batch[0], dict) and 'question' in batch[0]:
        # Handle streaming data
        input_texts = [f"Question: {item['question']}\n\nAnswer:" for item in batch]
        output_texts = [item['answer'] for item in batch]
        
        # Include CoT if available
        for i, item in enumerate(batch):
            if 'cot' in item and item['cot']:
                input_texts[i] = f"Question: {item['question']}\n\nChain of thought:"
                output_texts[i] = f"{item['cot']}\n\nAnswer: {item['answer']}"
        
        return {
            'input_texts': input_texts,
            'output_texts': output_texts
        }
    else:
        # Handle pre-processed data
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'output_ids': torch.stack([item['output_ids'] for item in batch]),
            'input_text': [item['input_text'] for item in batch],
            'output_text': [item['output_text'] for item in batch]
        }

class MMLUMedicalDataset(Dataset):
    """
    Handler for the HPAI-BSC/MMLU-medical-cot-llama31 dataset with memory efficiency.
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        streaming: bool = True
    ):
        """
        Initialize the dataset handler.
        
        Args:
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            streaming: Whether to stream the dataset to save memory
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.streaming = streaming
        self.dataset = None
        self.processed_data = None
        
        logger.info(f"MMLUMedicalDataset initialized with max_length: {max_length}")
    
    def load_data(self, split_ratio: float = 0.8) -> None:
        """
        Load the MMLU medical dataset from Hugging Face.
        
        Args:
            split_ratio: Train/validation split ratio
        """
        logger.info("Loading HPAI-BSC/MMLU-medical-cot-llama31 dataset")
        
        try:
            # Load the dataset
            dataset_name = "HPAI-BSC/MMLU-medical-cot-llama31"
            self.dataset = load_dataset(dataset_name, streaming=self.streaming)
            
            if not self.streaming:
                # Split into train and validation if not streaming
                train_size = int(len(self.dataset['train']) * split_ratio)
                val_size = len(self.dataset['train']) - train_size
                
                splits = self.dataset['train'].train_test_split(
                    train_size=train_size,
                    test_size=val_size,
                    seed=42
                )
                
                self.dataset = {
                    'train': splits['train'],
                    'validation': splits['test'],
                    'test': self.dataset.get('test', splits['test'])
                }
            
            logger.info("Dataset loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def prepare_for_distillation(self) -> None:
        """
        Process the dataset for knowledge distillation.
        
        This formats the data specifically for training with soft targets.
        """
        logger.info("Preparing dataset for knowledge distillation")
        
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        try:
            # Process the dataset for distillation
            if not self.streaming:
                self.processed_data = {
                    split: self._process_split(self.dataset[split])
                    for split in self.dataset
                }
            else:
                # For streaming datasets, we'll process on-the-fly
                logger.info("Using streaming mode - data will be processed on-the-fly")
                
            logger.info("Dataset prepared for distillation")
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            raise
    
    def _process_split(self, split_dataset) -> List[Dict[str, Any]]:
        """
        Process a dataset split for distillation.
        
        Args:
            split_dataset: Dataset split to process
            
        Returns:
            Processed data
        """
        processed_data = []
        
        for example in tqdm(split_dataset, desc="Processing dataset"):
            # Extract question and answer
            question = example.get('question', '')
            answer = example.get('answer', '')
            cot = example.get('cot', '')  # Chain-of-thought reasoning
            
            # Format input and output
            input_text = f"Question: {question}\n\nAnswer:"
            output_text = f"{answer}"
            
            # Include CoT if available
            if cot:
                input_text = f"Question: {question}\n\nChain of thought:"
                output_text = f"{cot}\n\nAnswer: {answer}"
            
            # Tokenize
            input_ids = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).input_ids.squeeze()
            
            output_ids = self.tokenizer(
                output_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).input_ids.squeeze()
            
            processed_data.append({
                'input_ids': input_ids,
                'output_ids': output_ids,
                'input_text': input_text,
                'output_text': output_text
            })
        
        return processed_data
    
    def create_memory_efficient_dataloader(
        self, 
        batch_size: int, 
        split: str = 'train',
        num_workers: int = 2
    ) -> DataLoader:
        """
        Create a memory-efficient DataLoader.
        """
        if self.streaming:
            return DataLoader(
                self.dataset[split],
                batch_size=batch_size,
                collate_fn=medical_dataset_collate_fn,  # Use the external collate function
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        else:
            if self.processed_data is None:
                raise ValueError(
                    "Data not processed. Call prepare_for_distillation() first."
                )
            
            return DataLoader(
                self.processed_data[split],
                batch_size=batch_size,
                shuffle=(split == 'train'),
                collate_fn=medical_dataset_collate_fn,  # Use the external collate function
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            Number of examples in the dataset
        """
        if self.streaming:
            # For streaming datasets, we don't know the length
            return 0
        
        if self.processed_data is not None:
            return len(self.processed_data['train'])
        
        if self.dataset is not None:
            return len(self.dataset['train'])
        
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing the item data
        """
        if self.streaming:
            raise ValueError("Cannot index into a streaming dataset")
        
        if self.processed_data is not None:
            return self.processed_data['train'][idx]
        
        raise ValueError("Data not processed. Call prepare_for_distillation() first.")


class VanillaKnowledgeDistillation:
    """
    Knowledge Distillation with soft targets and memory optimization.
    
    This implementation uses temperature scaling to soften probability distributions
    from the teacher model for the student to mimic.
    """
    
    def __init__(
        self, 
        teacher_model: PreTrainedModel, 
        student_model: PreTrainedModel,
        teacher_tokenizer: PreTrainedTokenizer,
        student_tokenizer: PreTrainedTokenizer,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ):
        """
        Initialize the knowledge distillation.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            teacher_tokenizer: Tokenizer for the teacher model
            student_tokenizer: Tokenizer for the student model
            temperature: Temperature for softening logits
            alpha: Weight for distillation loss vs. task loss
            device: Device to use for computation
            dtype: Data type for computation
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.dtype = dtype
        self.memory_manager = MemoryManager()
        
        # Ensure models are on the correct device
        if hasattr(self.teacher_model, 'device') and self.teacher_model.device != device:
            logger.warning(
                f"Teacher model is on {self.teacher_model.device}, "
                f"but distillation is configured for {device}"
            )
        
        if hasattr(self.student_model, 'device') and self.student_model.device != device:
            logger.warning(
                f"Student model is on {self.student_model.device}, "
                f"but distillation is configured for {device}"
            )
        
        logger.info(
            f"Knowledge Distillation initialized with temperature: {temperature}, "
            f"alpha: {alpha}, device: {device}, dtype: {dtype}"
        )
    
    def compute_distillation_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        grad_checkpoint: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the distillation loss with soft targets.
        
        Args:
            student_logits: Logits from the student model
            teacher_logits: Logits from the teacher model
            labels: Ground truth labels (optional)
            grad_checkpoint: Whether to use gradient checkpointing
            
        Returns:
            Tuple of (total loss, loss components dictionary)
        """
        # Apply temperature scaling
        soft_student_logits = student_logits / self.temperature
        soft_teacher_logits = teacher_logits / self.temperature
        
        # Compute KL divergence loss for soft targets
        # We use KL divergence between softmax distributions
        student_probs = torch.nn.functional.log_softmax(soft_student_logits, dim=-1)
        teacher_probs = torch.nn.functional.softmax(soft_teacher_logits, dim=-1)
        
        # Compute KL divergence loss
        distillation_loss = torch.nn.functional.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        ) * (self.temperature ** 2)
        
        # Initialize loss components dictionary
        loss_components = {'distillation_loss': distillation_loss.item()}
        
        # If labels are provided, compute task-specific loss
        if labels is not None:
            # Compute cross-entropy loss
            task_loss = torch.nn.functional.cross_entropy(
                student_logits, labels, reduction='mean'
            )
            loss_components['task_loss'] = task_loss.item()
            
            # Combine losses with alpha weighting
            total_loss = (
                self.alpha * distillation_loss + 
                (1 - self.alpha) * task_loss
            )
        else:
            # Use only distillation loss
            total_loss = distillation_loss
        
        loss_components['total_loss'] = total_loss.item()
        
        return total_loss, loss_components
    
    def train_step(
        self, 
        inputs: Dict[str, torch.Tensor],
        grad_accum_steps: int = 1,
        use_gradient_checkpointing: bool = True
    ) -> Dict[str, float]:
        """
        Perform a single training step for distillation.
        
        Args:
            inputs: Input tensors
            grad_accum_steps: Number of gradient accumulation steps
            use_gradient_checkpointing: Whether to use gradient checkpointing
            
        Returns:
            Dictionary of loss values
        """
        # Extract inputs
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        output_ids = inputs['output_ids'].to(self.device)
        output_mask = inputs.get('output_mask')
        if output_mask is not None:
            output_mask = output_mask.to(self.device)
        
        # Enable gradient checkpointing for student if requested
        if use_gradient_checkpointing and hasattr(self.student_model, 'gradient_checkpointing_enable'):
            self.student_model.gradient_checkpointing_enable()
        
        # Forward pass through teacher model (no gradients needed)
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            teacher_logits = teacher_outputs.logits
        
        # Forward pass through student model
        self.student_model.train()
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        student_logits = student_outputs.logits
        
        # Compute loss
        # For simplicity, we're using the last token's prediction for each sequence
        # In a real implementation, you might want to use all tokens or specific ones
        loss, loss_components = self.compute_distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            grad_checkpoint=use_gradient_checkpointing
        )
        
        # Scale loss for gradient accumulation
        if grad_accum_steps > 1:
            loss = loss / grad_accum_steps
        
        # Return loss components
        return loss_components


class DistillationTrainer:
    """
    Training manager for knowledge distillation with memory optimization.
    """
    
    def __init__(
        self, 
        distillation: VanillaKnowledgeDistillation,
        dataset: MMLUMedicalDataset,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_mixed_precision: bool = True,
        grad_accum_steps: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the distillation trainer.
        
        Args:
            distillation: Knowledge distillation instance
            dataset: Dataset handler
            optimizer: Optimizer for student model
            scheduler: Learning rate scheduler (optional)
            use_mixed_precision: Whether to use mixed precision training
            grad_accum_steps: Number of gradient accumulation steps
            device: Device to use for computation
        """
        self.distillation = distillation
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.grad_accum_steps = grad_accum_steps
        self.device = device
        self.memory_manager = MemoryManager()
        
        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        logger.info(
            f"DistillationTrainer initialized with "
            f"mixed precision: {self.use_mixed_precision}, "
            f"gradient accumulation steps: {self.grad_accum_steps}"
        )
    
    def train(
        self, 
        epochs: int,
        batch_size: int,
        checkpoint_dir: Optional[str] = None,
        eval_steps: int = 500,
        save_steps: int = 1000,
        max_grad_norm: float = 1.0,
        log_steps: int = 100,
        num_workers: int = 0  # Set to 0 to avoid multiprocessing issues
    ) -> Dict[str, List[float]]:
        """
        Train the student model with knowledge distillation.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            checkpoint_dir: Directory to save checkpoints
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between checkpoint saves
            max_grad_norm: Maximum gradient norm for clipping
            log_steps: Number of steps between logging
            num_workers: Number of worker processes for data loading
            
        Returns:
            Dictionary of training metrics
        """
        # Create checkpoint directory if needed
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Create dataloader with num_workers=0 to avoid multiprocessing issues
        train_dataloader = self.dataset.create_memory_efficient_dataloader(
            batch_size=batch_size,
            split='train',
            num_workers=num_workers
        )
        
        # Create validation dataloader if available
        try:
            val_dataloader = self.dataset.create_memory_efficient_dataloader(
                batch_size=batch_size,
                split='validation',
                num_workers=num_workers
            )
            has_validation = True
        except (KeyError, ValueError):
            has_validation = False
            logger.warning("No validation data available")
        
        # Initialize metrics
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0
            
            # Training
            self.distillation.student_model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                # Determine if this is the last accumulation step
                is_last_accum_step = (step + 1) % self.grad_accum_steps == 0
                
                # Mixed precision context
                with torch.cuda.amp.autocast() if self.use_mixed_precision else nullcontext():
                    # Forward and compute loss
                    loss_dict = self.distillation.train_step(
                        batch,
                        grad_accum_steps=self.grad_accum_steps,
                        use_gradient_checkpointing=(epoch == 0)  # Use checkpointing in first epoch
                    )
                    loss = torch.tensor(loss_dict['total_loss'], device=self.device)
                
                # Backward pass with mixed precision if enabled
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping and optimizer step on last accumulation step
                    if is_last_accum_step:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.distillation.student_model.parameters(),
                            max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        if self.scheduler:
                            self.scheduler.step()
                        self.optimizer.zero_grad()
                else:
                    # Standard backward pass
                    loss.backward()
                    
                    # Gradient clipping and optimizer step on last accumulation step
                    if is_last_accum_step:
                        torch.nn.utils.clip_grad_norm_(
                            self.distillation.student_model.parameters(),
                            max_grad_norm
                        )
                        self.optimizer.step()
                        if self.scheduler:
                            self.scheduler.step()
                        self.optimizer.zero_grad()
                
                # Update metrics
                epoch_loss += loss_dict['total_loss']
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss_dict['total_loss'],
                    'avg_loss': epoch_loss / (step + 1)
                })
                
                # Log metrics
                if global_step % log_steps == 0:
                    metrics['train_loss'].append(loss_dict['total_loss'])
                    if self.scheduler:
                        metrics['learning_rates'].append(self.scheduler.get_last_lr()[0])
                
                # Evaluate
                if has_validation and global_step % eval_steps == 0:
                    val_loss = self.evaluate(val_dataloader)
                    metrics['val_loss'].append(val_loss)
                    
                    # Save best model
                    if val_loss < best_val_loss and checkpoint_dir:
                        best_val_loss = val_loss
                        self.save_checkpoint(
                            os.path.join(checkpoint_dir, "best_model"),
                            global_step,
                            val_loss
                        )
                        logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
                
                # Save checkpoint
                if checkpoint_dir and global_step % save_steps == 0:
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, f"checkpoint-{global_step}"),
                        global_step,
                        loss_dict['total_loss']
                    )
                
                # Memory cleanup every 10 steps
                if global_step % 10 == 0:
                    self.memory_manager.cleanup()
            
            # End of epoch
            epoch_loss /= len(train_dataloader)
            logger.info(f"Epoch {epoch+1} completed with average loss: {epoch_loss:.4f}")
            
            # Evaluate at the end of each epoch
            if has_validation:
                val_loss = self.evaluate(val_dataloader)
                metrics['val_loss'].append(val_loss)
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss and checkpoint_dir:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, "best_model"),
                        global_step,
                        val_loss
                    )
                    logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
        
        logger.info("Training completed")
        return metrics
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate the student model on a validation dataset.
        
        Args:
            dataloader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.distillation.student_model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Forward pass with teacher and student
                loss_dict = self.distillation.train_step(
                    batch,
                    grad_accum_steps=1,
                    use_gradient_checkpointing=False
                )
                
                total_loss += loss_dict['total_loss']
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss
    
    def save_checkpoint(
        self, 
        path: str, 
        global_step: int, 
        loss: float,
        use_safetensors: bool = True
    ) -> None:
        """
        Save a checkpoint of the student model.
        
        Args:
            path: Path to save the checkpoint
            global_step: Current global step
            loss: Current loss value
            use_safetensors: Whether to use safetensors format
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.distillation.student_model.save_pretrained(
            path,
            safe_serialization=use_safetensors
        )
        
        # Save tokenizer
        self.distillation.student_tokenizer.save_pretrained(path)
        
        # Save training info
        with open(os.path.join(path, "training_info.json"), "w") as f:
            json.dump({
                "global_step": global_step,
                "loss": loss,
                "temperature": self.distillation.temperature,
                "alpha": self.distillation.alpha
            }, f)
        
        logger.info(f"Checkpoint saved to {path}")


class Evaluator:
    """
    Evaluation utilities for distilled models with perplexity and METEOR.
    """
    
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 8
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to use for computation
            batch_size: Batch size for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.memory_manager = MemoryManager(verbose=False)
        
        # Ensure model is in eval mode
        self.model.eval()
        
        logger.info(f"Evaluator initialized with batch_size: {batch_size}")
    
    def calculate_perplexity(
        self, 
        texts: List[str], 
        use_tqdm: bool = True
    ) -> float:
        """
        Calculate perplexity on evaluation data.
        
        Args:
            texts: List of text inputs
            use_tqdm: Whether to show progress bar
            
        Returns:
            Perplexity score (lower is better)
        """
        logger.info(f"Calculating perplexity on {len(texts)} texts")
        
        # Initialize
        total_loss = 0.0
        total_length = 0
        
        # Process in batches to avoid OOM
        iterator = range(0, len(texts), self.batch_size)
        if use_tqdm:
            iterator = tqdm(iterator, desc="Calculating perplexity")
        
        with torch.no_grad():
            for i in iterator:
                # Get batch
                batch_texts = texts[i:i+self.batch_size]
                
                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Move to device
                input_ids = encodings.input_ids.to(self.device)
                attention_mask = encodings.attention_mask.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                # Get loss
                loss = outputs.loss.item()
                
                # Update totals
                total_loss += loss * input_ids.size(0)
                total_length += input_ids.size(0)
                
                # Clean up memory
                del input_ids, attention_mask, outputs
                self.memory_manager.cleanup()
        
        # Calculate perplexity
        avg_loss = total_loss / total_length
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Perplexity: {perplexity:.4f}")
        
        return perplexity
    
    def calculate_meteor(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> float:
        """
        Calculate METEOR score for generated text.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            METEOR score (higher is better)
        """
        logger.info(f"Calculating METEOR score on {len(predictions)} samples")
        
        if len(predictions) != len(references):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) does not match "
                f"number of references ({len(references)})"
            )
        
        # Tokenize predictions and references
        tokenized_preds = [nltk.word_tokenize(pred.lower()) for pred in predictions]
        tokenized_refs = [nltk.word_tokenize(ref.lower()) for ref in references]
        
        # Calculate METEOR score for each sample
        meteor_scores = []
        
        for pred, ref in tqdm(zip(tokenized_preds, tokenized_refs), 
                             total=len(tokenized_preds),
                             desc="Calculating METEOR"):
            score = meteor_score([ref], pred)
            meteor_scores.append(score)
        
        # Calculate average METEOR score
        avg_meteor = sum(meteor_scores) / len(meteor_scores)
        
        logger.info(f"METEOR score: {avg_meteor:.4f}")
        
        return avg_meteor
    
    def evaluate_model(
        self, 
        test_data: List[Dict[str, str]],
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Run full evaluation with perplexity and METEOR.
        
        Args:
            test_data: List of test examples with 'input' and 'reference' keys
            max_samples: Maximum number of samples to evaluate (for memory constraints)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {len(test_data)} samples")
        
        # Limit samples if specified
        if max_samples and max_samples < len(test_data):
            logger.info(f"Limiting evaluation to {max_samples} samples")
            test_data = test_data[:max_samples]
        
        # Extract inputs and references
        inputs = [example['input'] for example in test_data]
        references = [example['reference'] for example in test_data]
        
        # Generate predictions
        predictions = []
        
        for i in tqdm(range(0, len(inputs), self.batch_size), desc="Generating predictions"):
            # Get batch
            batch_inputs = inputs[i:i+self.batch_size]
            
            # Tokenize
            encodings = self.tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            input_ids = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            # Decode
            batch_predictions = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
            # Add to predictions
            predictions.extend(batch_predictions)
            
            # Clean up memory
            del input_ids, attention_mask, outputs
            self.memory_manager.cleanup()
        
        # Calculate metrics
        perplexity = self.calculate_perplexity(inputs)
        meteor = self.calculate_meteor(predictions, references)
        
        # Return metrics
        metrics = {
            'perplexity': perplexity,
            'meteor': meteor
        }
        
        logger.info(f"Evaluation completed: {metrics}")
        
        return metrics


# Helper context manager for conditional contexts
class nullcontext:
    """Context manager that does nothing."""
    def __init__(self, enter_result=None):
        self.enter_result = enter_result
    
    def __enter__(self):
        return self.enter_result
    
    def __exit__(self, *excinfo):
        pass


"""
Example usage:

# 1. Load environment variables
api_key = load_env_variables()

# 2. Initialize model manager
model_manager = ModelManager(api_key=api_key)

# 3. Load teacher model (larger model)
teacher_model_name = "meta-llama/Llama-3-8b"
teacher_model = model_manager.load_model(
    model_name=teacher_model_name,
    is_teacher=True,
    use_8bit=True  # Use 8-bit quantization to save memory
)
teacher_tokenizer = model_manager.load_tokenizer(model_name=teacher_model_name)

# 4. Load student model (smaller model)
student_model_name = "meta-llama/Llama-3-1b"
student_model = model_manager.load_model(
    model_name=student_model_name,
    is_teacher=False
)
student_tokenizer = model_manager.load_tokenizer(model_name=student_model_name)

# 5. Initialize dataset
dataset = MMLUMedicalDataset(
    tokenizer=teacher_tokenizer,
    max_length=512,
    streaming=True  # Use streaming to save memory
)
dataset.load_data()
dataset.prepare_for_distillation()

# 6. Initialize optimizer and scheduler
optimizer = torch.optim.AdamW(
    student_model.parameters(),
    lr=5e-5,
    weight_decay=0.01
)

# Create scheduler with linear warmup
num_training_steps = 10000  # Adjust based on dataset size and epochs
num_warmup_steps = 1000
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 7. Initialize knowledge distillation
distillation = VanillaKnowledgeDistillation(
    teacher_model=teacher_model,
    student_model=student_model,
    teacher_tokenizer=teacher_tokenizer,
    student_tokenizer=student_tokenizer,
    temperature=2.0,  # Higher temperature for softer probabilities
    alpha=0.5  # Equal weight to distillation and task loss
)

# 8. Initialize trainer
trainer = DistillationTrainer(
    distillation=distillation,
    dataset=dataset,
    optimizer=optimizer,
    scheduler=scheduler,
    use_mixed_precision=True,  # Use mixed precision for memory efficiency
    grad_accum_steps=4  # Accumulate gradients for effective batch size of 4*batch_size
)

# 9. Train the model
metrics = trainer.train(
    epochs=3,
    batch_size=8,
    checkpoint_dir="./checkpoints",
    eval_steps=500,
    save_steps=1000
)

# 10. Evaluate the distilled model
evaluator = Evaluator(
    model=student_model,
    tokenizer=student_tokenizer,
    batch_size=16
)

# Prepare test data
test_data = [
    {
        'input': "Question: What is the most common cause of community-acquired pneumonia?",
        'reference': "The most common cause of community-acquired pneumonia is Streptococcus pneumoniae."
    },
    # Add more test examples...
]

# Run evaluation
eval_results = evaluator.evaluate_model(test_data)
print(f"Perplexity: {eval_results['perplexity']:.4f}")
print(f"METEOR score: {eval_results['meteor']:.4f}")

# Save the final distilled model
student_model.save_pretrained("./distilled_model")
student_tokenizer.save_pretrained("./distilled_model")
"""
