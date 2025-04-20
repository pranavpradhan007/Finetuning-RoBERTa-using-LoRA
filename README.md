# RoBERTa Fine-tuning with LoRA for AG News Classification

This script fine-tunes a `roberta-base` model on the AG News dataset for text classification using the Parameter-Efficient Fine-Tuning (PEFT) technique Low-Rank Adaptation (LoRA). It leverages the Hugging Face ecosystem (`transformers`, `datasets`, `peft`, `evaluate`) for data processing, model configuration, training, and evaluation.

## Overview

The script performs the following steps:

1.  **Loads** the AG News dataset.
2.  **Preprocesses** the text data using the RoBERTa tokenizer.
3.  **Splits** the data into training (90%) and validation (10%) sets.
4.  **Loads** the pre-trained `roberta-base` model for sequence classification.
5.  **Configures** LoRA (Low-Rank Adaptation) settings, targeting specific layers and modules with a rank pattern.
6.  **Applies** PEFT to the base model to create a LoRA-adapted model with significantly fewer trainable parameters.
7.  **Sets up** training using the Hugging Face `Trainer` API with specified `TrainingArguments` (optimizer, learning rate, epochs, batch size, evaluation strategy, etc.).
8.  **Includes** metrics computation (Accuracy, F1, Precision, Recall) and early stopping.
9.  **Trains** the LoRA-adapted model on the AG News training set, evaluating on the validation set.
10. **Evaluates** the best checkpoint found during training on the validation set.
11. **Provides** a function to perform inference on an external, unlabelled test dataset (loaded from a pickle file) and saves predictions to a CSV file.

## Features

*   Uses `roberta-base` as the foundation model.
*   Employs LoRA for parameter-efficient fine-tuning.
*   Targets specific modules (`query`, `value`, `output.dense`) in upper RoBERTa layers (8-11).
*   Uses a rank pattern for potentially different LoRA ranks per layer (`{ "8": 16, "9": 24, "10": 32, "11": 32 }`).
*   Utilizes the Hugging Face `Trainer` for streamlined training and evaluation.
*   Includes standard classification metrics calculation.
*   Implements early stopping based on validation accuracy.
*   Supports mixed-precision training (FP16) if CUDA is available.
*   Includes functionality for inference on a separate unlabelled test dataset.

## Requirements

The script requires the following Python libraries:

*   `transformers`
*   `datasets`
*   `evaluate`
*   `accelerate`
*   `peft`
*   `trl`
*   `bitsandbytes` (Optional, often used with PEFT/quantization, included in original pip install)
*   `torch`
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `tqdm`
*   `tensorboard` (for logging)
*   `ipywidgets` (for notebook progress bars)
*   `nvidia-ml-py3` (for GPU monitoring, potentially optional for basic execution)
*   `pickle` (Standard library)

You can install the primary dependencies using pip:

```bash
pip install transformers datasets evaluate accelerate peft trl bitsandbytes torch pandas numpy scikit-learn tqdm tensorboard ipywidgets nvidia-ml-py3
