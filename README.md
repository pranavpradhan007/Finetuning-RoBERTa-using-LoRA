# RoBERTa Fine-tuning with LoRA for AG News Classification (<1M Params)

This script fine-tunes a `roberta-base` model on the AG News dataset for text classification using the Parameter-Efficient Fine-Tuning (PEFT) technique Low-Rank Adaptation (LoRA). It adheres to a strict project constraint of **fewer than 1 million trainable parameters**. The script leverages the Hugging Face ecosystem (`transformers`, `datasets`, `peft`, `evaluate`) for data processing, model configuration, training, and evaluation.

## Overview

The script performs the following steps:

1.  **Loads** the AG News dataset.
2.  **Preprocesses** the text data using the RoBERTa tokenizer (padding to `max_length=512`).
3.  **Splits** the data into training (95%) and validation (5%) sets.
4.  **Loads** the pre-trained `roberta-base` model for sequence classification and **freezes** its parameters.
5.  **Calculates** the number of trainable parameters for a given LoRA configuration *before* applying it, using a helper function to ensure the <1M constraint is met.
6.  **Configures** LoRA settings (`r=8`, `alpha=32`, `bias="lora_only"`) targeting query, value, and specific feed-forward dense layers across the encoder.
7.  **Applies** PEFT to the frozen base model to create a LoRA-adapted model with the verified low number of trainable parameters (e.g., 907,012).
8.  **Sets up** training using the Hugging Face `Trainer` API with specified `TrainingArguments` (optimizer, learning rate=1e-4, epochs=10, batch size, etc.).
9.  **Includes** metrics computation (Accuracy only).
10. **Trains** the LoRA-adapted model on the AG News training set, evaluating on the validation set after each epoch.
11. **Loads** the best checkpoint found during training (based on validation accuracy).
12. **Evaluates** the best checkpoint on the validation set and prints final metrics and parameter counts.
13. **Provides** a function to perform inference on an external, unlabelled test dataset (loaded from a pickle file) and saves predictions to a CSV file.

## Features

*   Uses `roberta-base` as the foundation model.
*   Employs LoRA for parameter-efficient fine-tuning under a **<1M trainable parameter constraint**.
*   Calculates and verifies trainable parameters *before* model creation.
*   Targets specific modules (`query`, `value`, `roberta.encoder.layer.*.output.dense`) across all RoBERTa layers.
*   Uses LoRA configuration: `r=8`, `lora_alpha=32`, `lora_dropout=0.1`, `bias="lora_only"`.
*   Utilizes the Hugging Face `Trainer` for streamlined training and evaluation.
*   Computes Accuracy as the evaluation metric.
*   Saves the best model checkpoint based on validation accuracy.
*   Supports mixed-precision training (FP16) if CUDA is available.
*   Includes functionality for inference on a separate unlabelled test dataset.

## Requirements

The script requires the following Python libraries:

*   `transformers`
*   `datasets`
*   `evaluate`
*   `accelerate`
*   `peft`
*   `trl` (Included in original install)
*   `bitsandbytes` (Included in original install)
*   `torch`
*   `pandas`
*   `numpy`
*   `scikit-learn` (Used indirectly by `evaluate` or metrics)
*   `tqdm`
*   `matplotlib` (If plotting added/uncommented)
*   `pickle` (Standard library)
*   `tensorboard` (Optional, for logging if `report_to` is changed)
*   `ipywidgets` (Optional, for notebook progress bars)
*   `nvidia-ml-py3` (Optional, for GPU monitoring)

You can install the primary dependencies using pip:

```bash
pip install transformers datasets evaluate accelerate peft trl bitsandbytes torch pandas numpy scikit-learn tqdm matplotlib # Add other optionals if needed
