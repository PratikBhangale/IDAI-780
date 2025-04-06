#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Knowledge Distillation: Llama 3.2 3B → Llama 3.2 1B on MMLU-medical-cot-llama31 dataset

This script implements knowledge distillation from a teacher model (Llama 3.2 3B) 
to a student model (Llama 3.2 1B) using the MMLU-medical-cot-llama31 dataset.
"""

import os
import gc
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# Define the missing loss functions
def calculate_kl_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Calculate KL divergence loss between student and teacher logits.
    
    Args:
        student_logits: Logits from the student model
        teacher_logits: Logits from the teacher model
        temperature: Temperature for softening probability distributions
        
    Returns:
        KL divergence loss
    """
    if student_logits.size() != teacher_logits.size():
        # Handle size mismatch - truncate to smaller size
        min_size = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_size, :]
        teacher_logits = teacher_logits[:, :min_size, :]
    
    # Apply temperature scaling
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature
    
    # Apply softmax to get probabilities
    student_probs = F.softmax(student_logits_scaled, dim=-1)
    teacher_probs = F.softmax(teacher_logits_scaled, dim=-1)
    
    # Calculate KL divergence
    kl_div = F.kl_div(
        F.log_softmax(student_logits_scaled, dim=-1),
        teacher_probs,
        reduction='batchmean',
        log_target=False
    )
    
    # Scale by temperature squared as in the original paper
    return kl_div * (temperature ** 2)

def calculate_ce_loss(student_logits, target_ids, ignore_index=-100):
    """
    Calculate cross-entropy loss between student logits and target tokens.
    
    Args:
        student_logits: Logits from the student model
        target_ids: Target token IDs
        ignore_index: Token ID to ignore in loss calculation (usually pad token)
        
    Returns:
        Cross-entropy loss
    """
    # Shift logits and labels for next token prediction
    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    
    # Calculate cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    # Reshape for loss calculation
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup,
    set_seed
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("knowledge_distillation.log")
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
set_seed(42)

# Download necessary NLTK data
try:
    nltk.download('punkt')
    nltk.download('wordnet')
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")


class MedicalDataset(Dataset):
    """Dataset class for MMLU-medical-cot-llama31 dataset"""
    
    def __init__(self, dataset, max_length=512):
        self.dataset = dataset
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "system_prompt": item["system_prompt"],
            "question": item["question"],
            "response": item["response"]
        }


def format_prompt(system_prompt, question):
    """Format the input prompt by combining system prompt and question"""
    return f"{system_prompt}\n\nUser: {question}\n\nAssistant: "


def prepare_batch(batch_data, student_tokenizer, teacher_tokenizer, max_length=512, device="cuda"):
    """
    Prepare a batch of data for training with proper padding and attention masks.
    """
    # Extract data
    system_prompts = [item["system_prompt"] for item in batch_data]
    questions = [item["question"] for item in batch_data]
    responses = [item["response"] for item in batch_data]
    
    # Format prompts
    formatted_prompts = [format_prompt(sp, q) for sp, q in zip(system_prompts, questions)]
    
    # Tokenize inputs (formatted prompts) with padding for student model
    student_encoded_inputs = student_tokenizer(
        formatted_prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    # Tokenize targets (responses) with padding for student model
    student_encoded_targets = student_tokenizer(
        responses,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    # Tokenize inputs with teacher tokenizer if different
    if teacher_tokenizer != student_tokenizer:
        teacher_encoded_inputs = teacher_tokenizer(
            formatted_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
    else:
        teacher_encoded_inputs = student_encoded_inputs
    
    return {
        "student_input_ids": student_encoded_inputs.input_ids,
        "student_attention_mask": student_encoded_inputs.attention_mask,
        "teacher_input_ids": teacher_encoded_inputs.input_ids,
        "teacher_attention_mask": teacher_encoded_inputs.attention_mask,
        "target_input_ids": student_encoded_targets.input_ids,
        "target_attention_mask": student_encoded_targets.attention_mask
    }


def evaluate_model(model, tokenizer, dataset, max_length=512, max_new_tokens=50, batch_size=4, device="cuda"):
    """
    Evaluate model using METEOR score and perplexity metrics.
    """
    model.eval()
    total_meteor_score = 0
    total_perplexity = 0
    total = 0
    
    try:
        for i in range(0, len(dataset), batch_size):
            batch_data = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            
            for data in tqdm(batch_data, desc="Evaluating"):
                system_prompt = data["system_prompt"]
                question = data["question"]
                correct_answer = data["response"]
                
                # Format prompt
                prompt = format_prompt(system_prompt, question)
                
                # Generate model's response
                try:
                    inputs = tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=max_length
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=False
                        )
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract the generated response part
                    if "Assistant:" in generated_text:
                        generated_text = generated_text.split("Assistant:")[1].strip()
                    
                    # Calculate METEOR score
                    try:
                        reference_tokens = word_tokenize(correct_answer.lower())
                        candidate_tokens = word_tokenize(generated_text.lower())
                        
                        # Check for empty token lists
                        if len(reference_tokens) == 0 or len(candidate_tokens) == 0:
                            logger.warning("Empty token list detected, skipping METEOR calculation")
                            meteor = 0
                        else:
                            meteor = meteor_score([reference_tokens], candidate_tokens)
                        
                        total_meteor_score += meteor
                    except Exception as e:
                        logger.error(f"Error calculating METEOR score: {e}")
                        meteor = 0
                    
                    # Calculate perplexity
                    try:
                        target_encoding = tokenizer(
                            correct_answer, 
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_length
                        ).to(device)
                        
                        target_ids = target_encoding.input_ids
                        
                        with torch.no_grad():
                            outputs = model(input_ids=target_ids)
                            logits = outputs.logits
                            
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = target_ids[..., 1:].contiguous()
                        
                        loss_fct = torch.nn.CrossEntropyLoss(
                            ignore_index=tokenizer.pad_token_id, 
                            reduction='sum'
                        )
                        
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1)
                        )
                        
                        num_tokens = (shift_labels != tokenizer.pad_token_id).sum().item()
                        if num_tokens > 0:
                            try:
                                perplexity = torch.exp(loss / num_tokens).item()
                                # Check for NaN or Inf
                                if np.isnan(perplexity) or np.isinf(perplexity):
                                    logger.warning("Perplexity is NaN or Inf, using a high value instead")
                                    perplexity = 1e6  # Use a high value instead of NaN/Inf
                            except Exception as e:
                                logger.warning(f"Error calculating perplexity: {e}, using a high value instead")
                                perplexity = 1e6
                        else:
                            perplexity = 1e6  # Use a high value instead of Inf
                            logger.warning("No valid tokens for perplexity calculation")
                            
                        total_perplexity += perplexity
                    except Exception as e:
                        logger.error(f"Error calculating perplexity: {e}")
                        perplexity = float('inf')
                        
                    total += 1
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
                    continue
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
    
    # Calculate averages
    if total > 0:
        average_meteor_score = total_meteor_score / total
        average_perplexity = total_perplexity / total
    else:
        average_meteor_score = 0
        average_perplexity = float('inf')
        logger.warning("No examples were successfully evaluated")
    
    return average_meteor_score, average_perplexity


def get_teacher_logits(teacher_model, input_ids, attention_mask=None):
    """Get teacher logits for a batch"""
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        return teacher_outputs.logits


def plot_training_metrics(train_losses, eval_metrics, save_path="training_metrics.png"):
    """Plot training losses and evaluation metrics"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot METEOR scores
    epochs = range(1, len(eval_metrics['meteor']) + 1)
    ax2.plot(epochs, eval_metrics['meteor'])
    ax2.set_title('METEOR Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    
    # Plot perplexity
    ax3.plot(epochs, eval_metrics['perplexity'])
    ax3.set_title('Perplexity')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Perplexity')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    """Main function for knowledge distillation"""
    # Ensure using GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Check CUDA memory if available
    if torch.cuda.is_available():
        logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")
    
    # Get HuggingFace token from environment
    hf_token = os.environ.get("HF_TOKEN", None)
    if not hf_token:
        logger.warning("HF_TOKEN not found in environment variables. You may need this to access Meta-Llama models.")
    
    try:
        # Load the student model (Llama 3.2 1B)
        logger.info("Loading student model and tokenizer...")
        student_tokenizer = AutoTokenizer.from_pretrained(
            args.student_model,
            token=hf_token,
            use_fast=True,
            padding_side="right"  # Ensure consistent padding
        )
        
        # Set pad token if not defined
        if student_tokenizer.pad_token_id is None:
            logger.info("Setting pad token to EOS token for student tokenizer")
            student_tokenizer.pad_token = student_tokenizer.eos_token
            student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
        
        # Load student model with memory optimizations
        try:
            student_model = AutoModelForCausalLM.from_pretrained(
                args.student_model,
                token=hf_token,
                device_map="auto",  # Automatically determine best device mapping
                torch_dtype=torch.float16,  # Use half precision to save memory
                use_cache=False,  # Disable KV cache during training
                low_cpu_mem_usage=True  # Optimize CPU memory usage during loading
            )
            
            # Enable memory optimization techniques
            student_model.gradient_checkpointing_enable()  # Enable gradient checkpointing to save memory
            
            logger.info("Student model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading student model: {e}")
            # Try with more aggressive memory optimization if initial loading fails
            logger.info("Retrying with more aggressive memory optimization...")
            try:
                student_model = AutoModelForCausalLM.from_pretrained(
                    args.student_model,
                    token=hf_token,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    use_cache=False,
                    low_cpu_mem_usage=True,
                    offload_folder="student_offload_folder"  # Enable disk offloading
                )
                student_model.gradient_checkpointing_enable()
                logger.info("Student model loaded successfully with disk offloading")
            except Exception as e2:
                logger.error(f"Failed to load student model even with disk offloading: {e2}")
                raise
    except Exception as e:
        logger.error(f"Failed to load student model: {e}")
        raise
    
    try:
        # Load the teacher model (Llama 3.2 3B)
        logger.info("Loading teacher model and tokenizer...")
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_model,
            token=hf_token,
            use_fast=True,
            padding_side="right"  # Ensure consistent padding
        )
        
        # Set pad token if not defined
        if teacher_tokenizer.pad_token_id is None:
            logger.info("Setting pad token to EOS token for teacher tokenizer")
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
            teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
        
        # Load teacher model with memory optimizations
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            token=hf_token,
            device_map="auto",  # Automatically determine best device mapping
            torch_dtype=torch.float16,  # Use half precision to save memory
            low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
            offload_folder="offload_folder"  # Enable disk offloading if needed
        )
        
        # Set teacher model to evaluation mode and disable gradient computation
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
            
        logger.info("Teacher model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load teacher model: {e}")
        raise
    
    try:
        # Load the MMLU-medical-cot-llama31 dataset
        logger.info(f"Loading dataset: {args.dataset}")
        dataset = load_dataset(args.dataset, split="train")
        
        # Select a subset for faster experimentation if specified
        if args.subset_size > 0:
            subset = dataset.select(range(min(args.subset_size, len(dataset))))
        else:
            subset = dataset
            
        logger.info(f"Loaded {len(subset)} examples from {args.dataset}")
        
        # Display an example from the dataset
        example = subset[0]
        logger.info("\nExample input:")
        logger.info(f"System prompt: {example['system_prompt'][:100]}...")
        logger.info(f"Question: {example['question'][:100]}...")
        logger.info("\nExample response:")
        logger.info(f"{example['response'][:100]}...")
        
        # Create dataset object
        medical_dataset = MedicalDataset(subset, max_length=args.max_length)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Baseline evaluation
    try:
        logger.info("Evaluating student model before knowledge distillation...")
        initial_meteor, initial_perplexity = evaluate_model(
            student_model, 
            student_tokenizer, 
            medical_dataset, 
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.eval_batch_size,
            device=device
        )
        logger.info(f"Initial Student Model - Average METEOR Score: {initial_meteor * 100:.2f}%")
        logger.info(f"Initial Student Model - Average Perplexity: {initial_perplexity:.2f}")
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}")
        initial_meteor, initial_perplexity = 0, float('inf')
    
    # Create an optimizer for the student model
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate)
    
    # Create a learning rate scheduler
    total_steps = (len(medical_dataset) // args.batch_size) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Set up mixed precision training if requested
    try:
        if args.fp16:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            logger.info("Using mixed precision training")
        else:
            scaler = None
            # Define a dummy autocast context for non-fp16 training
            from contextlib import nullcontext
            autocast = lambda: nullcontext()
            logger.info("Using full precision training")
    except Exception as e:
        logger.error(f"Error setting up mixed precision training: {e}")
        logger.info("Falling back to full precision training")
        args.fp16 = False
        scaler = None
        from contextlib import nullcontext
        autocast = lambda: nullcontext()

    # Training loop with knowledge distillation
    logger.info("Starting knowledge distillation training...")
    
    # Track metrics
    train_losses = []
    eval_metrics = {'meteor': [], 'perplexity': []}
    
    try:
        for epoch in range(args.num_epochs):
            student_model.train()
            total_loss = 0
            batches = 0
            
            # Check if dataset is empty
            if len(medical_dataset) == 0:
                logger.error("Dataset is empty. Cannot proceed with training.")
                raise ValueError("Empty dataset")
                
            # Validate batch size
            if args.batch_size > len(medical_dataset):
                logger.warning(f"Batch size ({args.batch_size}) is larger than dataset size ({len(medical_dataset)}). Adjusting batch size.")
                args.batch_size = len(medical_dataset)

            # Create batches with progress bar
            progress_bar = tqdm(range(0, len(medical_dataset), args.batch_size), desc=f"Epoch {epoch+1}/{args.num_epochs}")
            
            for i in progress_bar:
                try:
                    batch_indices = list(range(i, min(i+args.batch_size, len(medical_dataset))))
                    if not batch_indices:
                        continue
                        
                    batch_data = [medical_dataset[j] for j in batch_indices]
                    
                    # Prepare batch data
                    batch = prepare_batch(
                        batch_data, 
                        student_tokenizer, 
                        teacher_tokenizer, 
                        max_length=args.max_length,
                        device=device
                    )

                    # Training step with improved error handling
                    try:
                        # Zero gradients at the beginning to avoid accumulation issues
                        optimizer.zero_grad()
                        
                        # Forward pass with mixed precision if enabled
                        if args.fp16:
                            with autocast():
                                # Get student model outputs
                                try:
                                    student_outputs = student_model(
                                        input_ids=batch["student_input_ids"],
                                        attention_mask=batch["student_attention_mask"]
                                    )
                                    student_logits = student_outputs.logits
                                except RuntimeError as e:
                                    if "CUDA out of memory" in str(e):
                                        logger.error("CUDA out of memory during student forward pass. Skipping batch.")
                                        torch.cuda.empty_cache()
                                        continue
                                    else:
                                        raise
                                
                                # Get teacher logits with no gradient
                                try:
                                    with torch.no_grad():
                                        teacher_logits = get_teacher_logits(
                                            teacher_model,
                                            batch["teacher_input_ids"],
                                            attention_mask=batch["teacher_attention_mask"]
                                        )
                                except RuntimeError as e:
                                    if "CUDA out of memory" in str(e):
                                        logger.error("CUDA out of memory during teacher forward pass. Skipping batch.")
                                        torch.cuda.empty_cache()
                                        continue
                                    else:
                                        raise
                                
                                # Calculate losses and handle NaN values
                                try:
                                    soft_loss = calculate_kl_loss(student_logits, teacher_logits, args.temperature)
                                    hard_loss = calculate_ce_loss(student_logits, batch["target_input_ids"])
                                    
                                    if torch.isnan(soft_loss) or torch.isinf(soft_loss):
                                        logger.warning("Soft loss is NaN or Inf, using zero instead")
                                        soft_loss = torch.zeros(1, device=device, requires_grad=True)
                                    
                                    if torch.isnan(hard_loss) or torch.isinf(hard_loss):
                                        logger.warning("Hard loss is NaN or Inf, using zero instead")
                                        hard_loss = torch.zeros(1, device=device, requires_grad=True)
                                    
                                    # Combined loss
                                    loss = (1 - args.alpha) * hard_loss + args.alpha * soft_loss
                                    
                                    # Check for NaN or Inf in combined loss
                                    if torch.isnan(loss) or torch.isinf(loss):
                                        logger.warning("Combined loss is NaN or Inf. Skipping batch.")
                                        continue
                                except Exception as e:
                                    logger.error(f"Error calculating loss: {e}")
                                    continue
                            
                            # Scale loss and backward pass
                            try:
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
                                scaler.step(optimizer)
                                scaler.update()
                            except RuntimeError as e:
                                if "CUDA out of memory" in str(e):
                                    logger.error("CUDA out of memory during backward pass. Skipping batch.")
                                    torch.cuda.empty_cache()
                                    continue
                                else:
                                    logger.error(f"Error in backward pass: {e}")
                                    continue
                            
                        else:
                            # Regular forward pass without mixed precision
                            try:
                                # Get student model outputs
                                try:
                                    student_outputs = student_model(
                                        input_ids=batch["student_input_ids"],
                                        attention_mask=batch["student_attention_mask"]
                                    )
                                    student_logits = student_outputs.logits
                                except RuntimeError as e:
                                    if "CUDA out of memory" in str(e):
                                        logger.error("CUDA out of memory during student forward pass. Skipping batch.")
                                        torch.cuda.empty_cache()
                                        continue
                                    else:
                                        raise
                                
                                # Get teacher logits with no gradient
                                try:
                                    with torch.no_grad():
                                        teacher_logits = get_teacher_logits(
                                            teacher_model,
                                            batch["teacher_input_ids"],
                                            attention_mask=batch["teacher_attention_mask"]
                                        )
                                except RuntimeError as e:
                                    if "CUDA out of memory" in str(e):
                                        logger.error("CUDA out of memory during teacher forward pass. Skipping batch.")
                                        torch.cuda.empty_cache()
                                        continue
                                    else:
                                        raise
                                
                                # Calculate losses and handle NaN values
                                try:
                                    soft_loss = calculate_kl_loss(student_logits, teacher_logits, args.temperature)
                                    hard_loss = calculate_ce_loss(student_logits, batch["target_input_ids"])
                                    
                                    if torch.isnan(soft_loss) or torch.isinf(soft_loss):
                                        logger.warning("Soft loss is NaN or Inf, using zero instead")
                                        soft_loss = torch.zeros(1, device=device, requires_grad=True)
                                    
                                    if torch.isnan(hard_loss) or torch.isinf(hard_loss):
                                        logger.warning("Hard loss is NaN or Inf, using zero instead")
                                        hard_loss = torch.zeros(1, device=device, requires_grad=True)
                                    
                                    # Combined loss
                                    loss = (1 - args.alpha) * hard_loss + args.alpha * soft_loss
                                    
                                    # Check for NaN or Inf in combined loss
                                    if torch.isnan(loss) or torch.isinf(loss):
                                        logger.warning("Combined loss is NaN or Inf. Skipping batch.")
                                        continue
                                except Exception as e:
                                    logger.error(f"Error calculating loss: {e}")
                                    continue
                                
                                # Regular backward pass
                                try:
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
                                    optimizer.step()
                                except RuntimeError as e:
                                    if "CUDA out of memory" in str(e):
                                        logger.error("CUDA out of memory during backward pass. Skipping batch.")
                                        torch.cuda.empty_cache()
                                        continue
                                    else:
                                        logger.error(f"Error in backward pass: {e}")
                                        continue
                            except Exception as e:
                                logger.error(f"Error in non-fp16 training path: {e}")
                                continue

                        # Common steps for both fp16 and regular training
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"NaN or Inf loss detected in batch {batches}. Skipping batch.")
                            optimizer.zero_grad()
                            continue

                        # Step the scheduler
                        try:
                            scheduler.step()
                        except Exception as e:
                            logger.error(f"Error stepping scheduler: {e}")
                        
                        # Make sure gradients are zeroed for next batch
                        optimizer.zero_grad()
                        
                        # Log memory usage periodically
                        if batches % 10 == 0 and torch.cuda.is_available():
                            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

                    except Exception as e:
                        logger.error(f"Error in training step: {e}")
                        optimizer.zero_grad()
                        continue
                    
                    total_loss += loss.item() * args.gradient_accumulation_steps
                    batches += 1
                    train_losses.append(loss.item() * args.gradient_accumulation_steps)
                    
                    # Update progress bar
                    if batches > 0:  # Add this check
                        progress_bar.set_postfix({
                            'loss': loss.item() * args.gradient_accumulation_steps,
                            'avg_loss': total_loss / batches
                        })
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    continue

            # Calculate average loss with safety check
            if batches > 0:
                avg_loss = total_loss / batches
                logger.info(f"Epoch {epoch+1}/{args.num_epochs} completed, Average Loss: {avg_loss:.4f}")
            else:
                logger.error("No batches were processed in this epoch")
                avg_loss = float('inf')

            # Evaluate after each epoch
            logger.info(f"Evaluating student model after epoch {epoch+1}...")
            meteor, perplexity = evaluate_model(
                student_model, 
                student_tokenizer, 
                medical_dataset, 
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.eval_batch_size,
                device=device
            )
            logger.info(f"Epoch {epoch+1} - METEOR Score: {meteor * 100:.2f}%")
            logger.info(f"Epoch {epoch+1} - Perplexity: {perplexity:.2f}")
            
            # Track metrics
            eval_metrics['meteor'].append(meteor)
            eval_metrics['perplexity'].append(perplexity)
            
            # Save checkpoint after each epoch
            try:
                checkpoint_path = checkpoint_dir / f"student_model_checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'meteor': meteor,
                    'perplexity': perplexity
                }, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
            
            # Plot and save training metrics
            plot_training_metrics(
                train_losses, 
                eval_metrics, 
                save_path=f"training_metrics_epoch_{epoch+1}.png"
            )
            
            # Clear cache to free up memory
            torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Final evaluation
    try:
        # Evaluate the student model after knowledge distillation
        logger.info("Evaluating student model after knowledge distillation...")
        final_meteor, final_perplexity = evaluate_model(
            student_model, 
            student_tokenizer, 
            medical_dataset, 
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.eval_batch_size,
            device=device
        )
        logger.info(f"Final Student Model - Average METEOR Score: {final_meteor * 100:.2f}%")
        logger.info(f"Final Student Model - Average Perplexity: {final_perplexity:.2f}")
        
        # Compare results
        logger.info("\nKnowledge Distillation Results Summary:")
        logger.info(f"METEOR Score: {initial_meteor * 100:.2f}% to {final_meteor * 100:.2f}%")
        logger.info(f"Perplexity: {initial_perplexity:.2f} to {final_perplexity:.2f}")
        
        # Calculate improvement with NaN handling
        if initial_meteor > 0 and not np.isnan(final_meteor) and not np.isnan(initial_meteor):
            meteor_improvement = (final_meteor - initial_meteor) / initial_meteor * 100
        else:
            meteor_improvement = 0
            
        if initial_perplexity > 0 and not np.isnan(final_perplexity) and not np.isnan(initial_perplexity) and not np.isinf(initial_perplexity) and not np.isinf(final_perplexity):
            perplexity_improvement = (initial_perplexity - final_perplexity) / initial_perplexity * 100
        else:
            perplexity_improvement = 0
        
        logger.info(f"METEOR Score Improvement: {meteor_improvement:.2f}%")
        logger.info(f"Perplexity Improvement: {perplexity_improvement:.2f}%")
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
    
    # Save the final distilled model
    try:
        logger.info("Saving the final distilled model...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        student_model.save_pretrained(output_dir / "distilled_model")
        student_tokenizer.save_pretrained(output_dir / "distilled_model")
        
        logger.info(f"Distilled model saved to {output_dir / 'distilled_model'}")
    except Exception as e:
        logger.error(f"Failed to save distilled model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Llama 3.2 models")
    
    # Model parameters
    parser.add_argument("--student_model", type=str, default="meta-llama/Llama-3.2-1B", 
                        help="Path or name of the student model")
    parser.add_argument("--teacher_model", type=str, default="meta-llama/Llama-3.2-3B", 
                        help="Path or name of the teacher model")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="HPAI-BSC/MMLU-medical-cot-llama31", 
                        help="Dataset name or path")
    parser.add_argument("--subset_size", type=int, default=100, 
                        help="Number of examples to use (0 for all)")
    parser.add_argument("--max_length", type=int, default=512, 
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--max_new_tokens", type=int, default=50, 
                        help="Maximum number of new tokens to generate during evaluation")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, 
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=100, 
                        help="Number of warmup steps for learning rate scheduler")
    
    # Knowledge distillation parameters
    parser.add_argument("--alpha", type=float, default=0.5, 
                        help="Weight for KL divergence loss (0-1)")
    parser.add_argument("--temperature", type=float, default=2.0, 
                        help="Temperature for softening probability distributions")
    
    # Output parameters
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                        help="Directory to save checkpoints")
    parser.add_argument("--output_dir", type=str, default="distilled_models", 
                        help="Directory to save the final model")
    
    # Add to the argument parser section
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--loss_scale", type=float, default=1.0,
                        help="Loss scaling factor for numerical stability")
    
    args = parser.parse_args()
    
    main(args)
