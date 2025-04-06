# Knowledge Distillation for Llama 3.2 Models

This script implements knowledge distillation from a teacher model (Llama 3.2 3B) to a student model (Llama 3.2 1B) using the MMLU-medical-cot-llama31 dataset.

## Overview

Knowledge distillation is a technique where a smaller model (student) is trained to mimic the behavior of a larger model (teacher). This approach allows the student model to achieve better performance than if it were trained from scratch, while maintaining a smaller size and faster inference time.

## Features

- Implements KL divergence loss for soft targets from teacher model
- Uses cross-entropy loss for hard targets from ground truth
- Supports mixed precision training (fp16) for memory efficiency
- Includes robust error handling for CUDA out-of-memory errors
- Implements gradient checkpointing and model offloading for large models
- Evaluates models using METEOR score and perplexity metrics
- Saves checkpoints after each epoch
- Generates training metrics plots

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- NLTK
- Datasets
- Matplotlib
- HuggingFace account with access to Meta-Llama models

## Installation

```bash
# Create a virtual environment
conda create -n knowledge_distillation python=3.8
conda activate knowledge_distillation

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets nltk matplotlib tqdm
```

## Usage

### Setting up HuggingFace Token

To access Meta-Llama models, you need to set your HuggingFace token as an environment variable:

```bash
export HF_TOKEN=your_huggingface_token
```

### Running the Script

Basic usage:

```bash
python llama_medical_kd.py
```

With custom parameters:

```bash
python llama_medical_kd.py \
  --student_model meta-llama/Llama-3.2-1B \
  --teacher_model meta-llama/Llama-3.2-3B \
  --dataset HPAI-BSC/MMLU-medical-cot-llama31 \
  --subset_size 100 \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --alpha 0.5 \
  --temperature 2.0 \
  --fp16
```

### Command Line Arguments

#### Model Parameters
- `--student_model`: Path or name of the student model (default: "meta-llama/Llama-3.2-1B")
- `--teacher_model`: Path or name of the teacher model (default: "meta-llama/Llama-3.2-3B")

#### Dataset Parameters
- `--dataset`: Dataset name or path (default: "HPAI-BSC/MMLU-medical-cot-llama31")
- `--subset_size`: Number of examples to use, 0 for all (default: 100)
- `--max_length`: Maximum sequence length for tokenization (default: 512)
- `--max_new_tokens`: Maximum number of new tokens to generate during evaluation (default: 50)

#### Training Parameters
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 4)
- `--eval_batch_size`: Evaluation batch size (default: 4)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 4)
- `--warmup_steps`: Number of warmup steps for learning rate scheduler (default: 100)

#### Knowledge Distillation Parameters
- `--alpha`: Weight for KL divergence loss (0-1) (default: 0.5)
- `--temperature`: Temperature for softening probability distributions (default: 2.0)

#### Output Parameters
- `--checkpoint_dir`: Directory to save checkpoints (default: "checkpoints")
- `--output_dir`: Directory to save the final model (default: "distilled_models")

#### Additional Parameters
- `--max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)
- `--fp16`: Use mixed precision training (default: False)
- `--loss_scale`: Loss scaling factor for numerical stability (default: 1.0)

## Memory Optimization

The script includes several memory optimization techniques:

1. **Gradient Checkpointing**: Reduces memory usage by recomputing intermediate activations during the backward pass.
2. **Mixed Precision Training**: Uses fp16 to reduce memory usage and speed up training.
3. **Model Offloading**: Can offload parts of the model to disk if needed.
4. **Robust Error Handling**: Handles CUDA out-of-memory errors gracefully.

## Outputs

The script produces the following outputs:

1. **Checkpoints**: Saved after each epoch in the `checkpoints` directory.
2. **Training Metrics**: Plots of training loss, METEOR score, and perplexity saved as PNG files.
3. **Distilled Model**: The final distilled model saved in the `distilled_models` directory.
4. **Logs**: Detailed logs saved in `knowledge_distillation.log`.

## Troubleshooting

### CUDA Out of Memory Errors

If you encounter CUDA out-of-memory errors, try the following:

1. Reduce the batch size (`--batch_size`)
2. Enable mixed precision training (`--fp16`)
3. Reduce the maximum sequence length (`--max_length`)
4. Use a smaller subset of the dataset (`--subset_size`)

### Model Loading Issues

If you have issues loading the models, ensure you have:

1. Set the `HF_TOKEN` environment variable
2. Have access to the Meta-Llama models on HuggingFace
3. Have a stable internet connection

## License

This project is licensed under the MIT License - see the LICENSE file for details.
