#!/usr/bin/env python3
"""
Flexible Model Embedding Analysis

This script computes delta embeddings for any list of HuggingFace models.
It's designed for researchers who want to analyze their own fine-tuned models
or reproduce specific experiments from the paper.

The script takes a base model and list of fine-tuned models, then computes
delta embeddings (fine-tuned - base) using generic probes.

Usage Examples:

# Custom models
python compute_embeddings.py \
    --base_model "meta-llama/Llama-3.1-8B" \
    --models "user/model1" "user/model2" "user/model3" \
    --output_file "my_embeddings.npz"

# Reproduce Llama experiment
python compute_embeddings.py \
    --base_model "meta-llama/Llama-3.1-8B" \
    --models \
        "ASethi04/llama-3.1-8b-legalbench-first" \
        "ASethi04/llama-3.1-8b-legalbench-second" \
        "ASethi04/llama-3.1-8b-legalbench-third" \
        "ASethi04/llama-3.1-8b-gsm8k-first" \
        "ASethi04/llama-3.1-8b-gsm8k-second" \
        "ASethi04/llama-3.1-8b-gsm8k-third" \
        "ASethi04/llama-3.1-8b-pubmedqa-first" \
        "ASethi04/llama-3.1-8b-pubmedqa-second" \
        "ASethi04/llama-3.1-8b-pubmedqa-third" \
        "ASethi04/llama-3.1-8b-hellaswag-first" \
        "ASethi04/llama-3.1-8b-hellaswag-second" \
        "ASethi04/llama-3.1-8b-hellaswag-third" \
        "ASethi04/llama-3.1-8b-opc-first" \
        "ASethi04/llama-3.1-8b-opc-second" \
        "ASethi04/llama-3.1-8b-opc-third" \
    --output_file "llama_embeddings.npz"

Requirements:
    pip install torch transformers peft numpy tqdm huggingface_hub matplotlib seaborn scikit-learn
"""

import argparse
import contextlib
import gc
import logging
import os
from typing import Dict, List

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_dataset import get_formatted_probes, get_probe_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute delta embeddings for fine-tuned models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Commands:

# Reproduce Llama experiment from paper:
python compute_embeddings.py \\
    --base_model "meta-llama/Llama-3.1-8B" \\
    --models "ASethi04/llama-3.1-8b-legalbench-first" \\
             "ASethi04/llama-3.1-8b-gsm8k-first" \\
             "ASethi04/llama-3.1-8b-pubmedqa-first" \\
    --output_file "llama_embeddings.npz"

# Reproduce Qwen experiment from paper:
python compute_embeddings.py \\
    --base_model "Qwen/Qwen2.5-7B" \\
    --models "ASethi04/qwen-2.5-7b-legalbench-first" \\
             "ASethi04/qwen-2.5-7b-gsm8k-first" \\
             "ASethi04/qwen-2.5-7b-pubmedqa-first" \\
    --output_file "qwen_embeddings.npz"

# Reproduce Gemma experiment from paper:
python compute_embeddings.py \\
    --base_model "google/gemma-2-9b" \\
    --models "ASethi04/gemma-2-9b-legalbench-first" \\
             "ASethi04/gemma-2-9b-gsm8k-first" \\
             "ASethi04/gemma-2-9b-pubmedqa-first" \\
    --output_file "gemma_embeddings.npz"

# Analyze your own models:
python compute_embeddings.py \\
    --base_model "your_org/base_model" \\
    --models "your_org/finetuned_model_1" "your_org/finetuned_model_2" \\
    --output_file "my_embeddings.npz"
        """
    )
    
    parser.add_argument(
        "--base_model",
        required=True,
        help="Base model name (HuggingFace model ID)"
    )
    parser.add_argument(
        "--models",
        nargs='+',
        required=True,
        help="List of fine-tuned model names (HuggingFace model IDs)"
    )
    parser.add_argument(
        "--output_file",
        default="embeddings.npz",
        help="Output file to save embeddings (.npz format) (default: embeddings.npz)"
    )
    parser.add_argument(
        "--cuda_devices",
        default="0",
        help="CUDA device to use (default: 0)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization (default: 256)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def log_memory_usage():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


@contextlib.contextmanager
def load_model_temporarily(model_path, base_model, tokenizer, device):
    """
    Context manager to load and automatically cleanup models.
    
    Ensures proper memory management when loading multiple models sequentially.
    """
    model = None
    additional_base = None
    
    try:
        logger.info(f"Loading model: {model_path}")
        peft_config = PeftConfig.from_pretrained(model_path)
        peft_base_model_name = peft_config.base_model_name_or_path
        
        # Check if we need a different base model
        if base_model is None or peft_base_model_name != base_model.config.name_or_path:
            logger.info(f"Loading base model for PEFT: {peft_base_model_name}")
            additional_base = AutoModelForCausalLM.from_pretrained(
                peft_base_model_name, 
                device_map="auto"
            )
            model = PeftModel.from_pretrained(additional_base, model_path)
        else:
            model = PeftModel.from_pretrained(base_model, model_path)
        
        model.config.output_hidden_states = True
        model.eval()
        yield model
        
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        raise
    finally:
        # Clean up memory - order matters for efficiency
        if model is not None:
            del model
        if additional_base is not None:
            del additional_base
        torch.cuda.empty_cache()
        gc.collect()


def get_average_activation(model, probe_texts, tokenizer, device, max_length=256):
    """
    Compute the average activation from a model for given probe texts.
    
    Args:
        model: The model to analyze
        probe_texts: List of probe strings
        tokenizer: Model tokenizer
        device: Device to run inference on
        max_length: Maximum sequence length
        
    Returns:
        numpy.ndarray: Average activation vector
    """
    model.eval()
    activations = []
    
    for text in tqdm(probe_texts, desc="Computing activations", leave=False):
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Extract last layer hidden state and last token representation
        hidden = outputs.hidden_states[-1].float()
        last_hidden = hidden[:, -1, :].squeeze(0).cpu().numpy()
        activations.append(last_hidden)
        
    return np.mean(np.stack(activations), axis=0)


def compute_embeddings(base_model_name: str, model_paths: List[str], 
                      cuda_devices: str = "0", max_length: int = 256,
                      verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute delta embeddings for a list of fine-tuned models.
    
    Args:
        base_model_name: HuggingFace model ID for base model
        model_paths: List of HuggingFace model IDs for fine-tuned models
        cuda_devices: CUDA device specification
        max_length: Maximum sequence length for tokenization
        verbose: Enable verbose logging
        
    Returns:
        Dict mapping model names to their delta embeddings
    """
    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if verbose:
        log_memory_usage()
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cuda")
    base_model.config.output_hidden_states = True
    base_model.eval()
    
    if verbose:
        log_memory_usage()
    
    # Get probe texts
    probe_texts = get_formatted_probes()
    probe_info = get_probe_info()
    logger.info(f"Using {probe_info['num_templates']} probe templates")
    
    # Compute base model activation
    logger.info("Computing base model activations...")
    base_activation = get_average_activation(
        base_model, probe_texts, tokenizer, device, max_length
    )
    
    # Process each fine-tuned model
    logger.info(f"Processing {len(model_paths)} fine-tuned models...")
    embeddings = {}
    
    for i, model_path in enumerate(model_paths):
        logger.info(f"Processing model {i+1}/{len(model_paths)}: {model_path}")
        
        try:
            with load_model_temporarily(model_path, base_model, tokenizer, device) as model:
                # Compute fine-tuned model activation
                finetuned_activation = get_average_activation(
                    model, probe_texts, tokenizer, device, max_length
                )
                
                # Compute delta embedding (fine-tuned - base)
                delta_embedding = finetuned_activation - base_activation
                embeddings[model_path] = delta_embedding
                
                logger.info(f"Successfully processed {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to process {model_path}: {e}")
            continue
        
        if verbose:
            log_memory_usage()
    
    # Clean up base model
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    if verbose:
        log_memory_usage()
    
    logger.info(f"Successfully computed embeddings for {len(embeddings)}/{len(model_paths)} models")
    return embeddings


def save_embeddings(embeddings: Dict[str, np.ndarray], output_file: str, 
                   base_model_name: str):
    """
    Save embeddings to file with metadata.
    
    Args:
        embeddings: Dictionary of model names to embeddings
        output_file: Output file path
        base_model_name: Base model used for computation
    """
    # Prepare data for saving
    model_names = list(embeddings.keys())
    embedding_matrix = np.stack([embeddings[name] for name in model_names])
    
    # Save with metadata
    np.savez(
        output_file,
        embeddings=embedding_matrix,
        model_names=model_names,
        base_model=base_model_name,
        probe_info=get_probe_info()
    )
    
    logger.info(f"Saved embeddings to {output_file}")
    logger.info(f"Shape: {embedding_matrix.shape}")
    logger.info(f"Models: {len(model_names)}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Print configuration
    logger.info("="*60)
    logger.info("Model Embedding Computation")
    logger.info("="*60)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Fine-tuned models: {len(args.models)}")
    logger.info(f"Output file: {args.output_file}")
    logger.info("="*60)
    
    # Compute embeddings
    embeddings = compute_embeddings(
        base_model_name=args.base_model,
        model_paths=args.models,
        cuda_devices=args.cuda_devices,
        max_length=args.max_length,
        verbose=args.verbose
    )
    
    if len(embeddings) == 0:
        logger.error("No embeddings computed successfully. Exiting.")
        return
    
    # Save results
    save_embeddings(embeddings, args.output_file, args.base_model)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPUTATION COMPLETE")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"Successfully processed: {len(embeddings)}/{len(args.models)} models")
    print(f"Embedding dimension: {list(embeddings.values())[0].shape[0]}")
    print(f"Results saved to: {args.output_file}")
    print("="*60)
    
    # Show sample usage for loading results
    print("\nTo load results in Python:")
    print(f"import numpy as np")
    print(f"data = np.load('{args.output_file}')")
    print(f"embeddings = data['embeddings']")
    print(f"model_names = data['model_names']")
    print(f"print('Loaded embeddings shape:', embeddings.shape)")


if __name__ == "__main__":
    main()