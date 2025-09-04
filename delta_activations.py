#!/usr/bin/env python3
"""
Model Embedding Analysis: Reproducing Task-Specific Fine-tuning Results

This script reproduces the model embedding analysis from the paper, analyzing how 
fine-tuned language models cluster by task across different model families.

The script computes delta embeddings (fine-tuned - base model activations) and 
evaluates clustering quality using silhouette analysis and t-SNE visualization.

Usage:
    python delta_activations.py --model llama
    python delta_activations.py --model qwen  
    python delta_activations.py --model gemma

Requirements:
    pip install torch transformers peft sklearn matplotlib seaborn numpy tqdm huggingface_hub
"""

import argparse
import contextlib
import gc
import json
import logging
import os
import tempfile
from collections import Counter
from typing import Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from peft import PeftConfig, PeftModel
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce model embedding analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python delta_activations.py --model llama
    python delta_activations.py --model qwen
    python delta_activations.py --model gemma
        """
    )
    parser.add_argument(
        "--model", 
        choices=["llama", "qwen", "gemma"], 
        required=True,
        help="Model family to analyze (llama, qwen, or gemma)"
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory to save results (default: ./results)"
    )
    parser.add_argument(
        "--cuda_devices", 
        default="",
        help="CUDA device to use (default: none)"
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
    
    This ensures proper memory management when loading multiple models sequentially.
    """
    model = None
    additional_base = None
    
    try:
        logger.info(f"Loading model: {model_path}")
        peft_config = PeftConfig.from_pretrained(model_path)
        peft_base_model_name = peft_config.base_model_name_or_path
        
        # Check if we need a different base model
        if base_model is None or peft_base_model_name != base_model.config.name_or_path:
            logger.info(f"Loading base model: {peft_base_model_name}")
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


def get_model_config(model_family):
    """
    Get the model configuration for each family.
    
    Returns the base model name and list of fine-tuned model paths.
    These are the clean, renamed models from the paper.
    """
    configs = {
        "llama": {
            "base_model": "meta-llama/Llama-3.1-8B",
            "models": [
                # LegalBench models (3 splits)
                "ASethi04/llama-3.1-8b-legalbench-first",
                "ASethi04/llama-3.1-8b-legalbench-second", 
                "ASethi04/llama-3.1-8b-legalbench-third",
                # GSM8K models (3 splits)
                "ASethi04/llama-3.1-8b-gsm8k-first",
                "ASethi04/llama-3.1-8b-gsm8k-second",
                "ASethi04/llama-3.1-8b-gsm8k-third",
                # PubMedQA models (3 splits) 
                "ASethi04/llama-3.1-8b-pubmedqa-first",
                "ASethi04/llama-3.1-8b-pubmedqa-second",
                "ASethi04/llama-3.1-8b-pubmedqa-third",
                # HellaSwag models (3 splits)
                "ASethi04/llama-3.1-8b-hellaswag-first",
                "ASethi04/llama-3.1-8b-hellaswag-second",
                "ASethi04/llama-3.1-8b-hellaswag-third",
                # OPC models (3 splits)
                "ASethi04/llama-3.1-8b-opc-first",
                "ASethi04/llama-3.1-8b-opc-second",
                "ASethi04/llama-3.1-8b-opc-third"
            ]
        },
        "qwen": {
            "base_model": "Qwen/Qwen2.5-7B",
            "models": [
                # LegalBench models (3 splits)
                "ASethi04/qwen-2.5-7b-legalbench-first",
                "ASethi04/qwen-2.5-7b-legalbench-second",
                "ASethi04/qwen-2.5-7b-legalbench-third",
                # GSM8K models (3 splits)
                "ASethi04/qwen-2.5-7b-gsm8k-first",
                "ASethi04/qwen-2.5-7b-gsm8k-second",
                "ASethi04/qwen-2.5-7b-gsm8k-third",
                # PubMedQA models (3 splits)
                "ASethi04/qwen-2.5-7b-pubmedqa-first",
                "ASethi04/qwen-2.5-7b-pubmedqa-second",
                "ASethi04/qwen-2.5-7b-pubmedqa-third",
                # HellaSwag models (3 splits)
                "ASethi04/qwen-2.5-7b-hellaswag-first",
                "ASethi04/qwen-2.5-7b-hellaswag-second",
                "ASethi04/qwen-2.5-7b-hellaswag-third",
                # OPC models (3 splits)
                "ASethi04/qwen-2.5-7b-opc-first",
                "ASethi04/qwen-2.5-7b-opc-second",
                "ASethi04/qwen-2.5-7b-opc-third"
            ]
        },
        "gemma": {
            "base_model": "google/gemma-2-9b", 
            "models": [
                # LegalBench models (3 splits)
                "ASethi04/gemma-2-9b-legalbench-first",
                "ASethi04/gemma-2-9b-legalbench-second",
                "ASethi04/gemma-2-9b-legalbench-third",
                # GSM8K models (3 splits)
                "ASethi04/gemma-2-9b-gsm8k-first",
                "ASethi04/gemma-2-9b-gsm8k-second",
                "ASethi04/gemma-2-9b-gsm8k-third",
                # PubMedQA models (3 splits)
                "ASethi04/gemma-2-9b-pubmedqa-first",
                "ASethi04/gemma-2-9b-pubmedqa-second",
                "ASethi04/gemma-2-9b-pubmedqa-third",
                # HellaSwag models (3 splits)
                "ASethi04/gemma-2-9b-hellaswag-first",
                "ASethi04/gemma-2-9b-hellaswag-second",
                "ASethi04/gemma-2-9b-hellaswag-third",
                # OPC models (3 splits)
                "ASethi04/gemma-2-9b-opc-first",
                "ASethi04/gemma-2-9b-opc-second",
                "ASethi04/gemma-2-9b-opc-third"
            ]
        }
    }
    return configs[model_family]


def create_generic_probes():
    """
    Create the generic probe templates used in the paper.
    
    These probes are task-agnostic and designed to elicit general model behavior
    across different fine-tuned models.
    """
    probe_templates = [
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "The task described below requires a response that completes the request accurately.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "Below is a description of a task. Provide a response that aligns with the requirements.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "The following instruction outlines a task. Generate a response that meets the specified request.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "You are given an instruction and input. Write a response that completes the task as requested.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:"
    ]
    
    # Create probe texts with generic task and input
    task_prompt = "Please provide a response."
    input_text = "Input."
    formatted_probes = [
        template.format(task=task_prompt, input=input_text) 
        for template in probe_templates
    ]
    
    logger.info(f"Created {len(formatted_probes)} generic probe templates")
    return formatted_probes


def get_average_activation(model, texts, tokenizer, device):
    """
    Compute the average activation from a model for given probe texts.
    
    This extracts the last layer hidden states and averages them across
    all probe texts to get a representative embedding.
    """
    model.eval()
    activations = []
    
    for text in tqdm(texts, desc="Computing activations", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Extract last layer hidden state and last token representation
        hidden = outputs.hidden_states[-1].float()
        last_hidden = hidden[:, -1, :].squeeze(0).cpu().numpy()
        activations.append(last_hidden)
        
    return np.mean(np.stack(activations), axis=0)


def categorize_models(model_paths):
    """
    Categorize models by task type based on their names.
    
    Returns task labels for clustering analysis.
    """
    labels = []
    task_names = []
    
    # Define task categories
    task_mapping = {
        "legalbench": (0, "LegalBench"),
        "gsm8k": (1, "GSM8K"), 
        "pubmedqa": (2, "PubMedQA"),
        "hellaswag": (3, "HellaSwag"),
        "opc": (4, "OPC-SFT")
    }
    
    for model_path in model_paths:
        model_name = model_path.lower()
        
        # Find which task this model belongs to
        task_found = False
        for task_key, (label, task_name) in task_mapping.items():
            if task_key in model_name:
                labels.append(label)
                task_names.append(task_name)
                task_found = True
                break
        
        if not task_found:
            labels.append(5)  # Unknown category
            task_names.append("Unknown")
    
    return labels, task_names


def compute_silhouette_analysis(embeddings_dict, output_dir):
    """
    Compute and visualize silhouette analysis for task clustering.
    
    This measures how well models cluster by their training task.
    """
    if len(embeddings_dict) <= 2:
        logger.warning("Need at least 3 models for silhouette analysis")
        return None
    
    logger.info("Computing silhouette analysis...")
    
    model_names = list(embeddings_dict.keys())
    embeddings_matrix = np.stack([embeddings_dict[name] for name in model_names])
    
    # Compute cosine distance matrix
    similarity_matrix = cosine_similarity(embeddings_matrix)
    distance_matrix = np.clip(1 - similarity_matrix, 0, 1)
    
    # Get task labels
    labels, task_names_list = categorize_models(model_names)
    
    # Filter out tasks with only one model
    label_counts = Counter(labels)
    valid_labels = [label for label, count in label_counts.items() if count >= 2]
    
    if len(valid_labels) < 2:
        logger.warning("Need at least 2 tasks with 2+ models each")
        return None
    
    # Filter to valid samples
    valid_indices = [i for i, label in enumerate(labels) if label in valid_labels]
    filtered_distance_matrix = distance_matrix[valid_indices][:, valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    
    # Remap labels to be consecutive
    unique_labels = sorted(set(filtered_labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = [label_map[label] for label in filtered_labels]
    
    # Compute silhouette scores
    avg_score = silhouette_score(filtered_distance_matrix, remapped_labels, metric="precomputed")
    individual_scores = silhouette_samples(filtered_distance_matrix, remapped_labels, metric="precomputed")
    
    logger.info(f"Silhouette Score: {avg_score:.4f}")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    y_lower = 10
    
    task_names_mapping = ["LegalBench", "GSM8K", "PubMedQA", "HellaSwag", "OPC-SFT", "Unknown"]
    
    for i, label in enumerate(range(len(unique_labels))):
        cluster_scores = [score for score, lbl in zip(individual_scores, remapped_labels) if lbl == label]
        if not cluster_scores:
            continue
            
        cluster_scores.sort()
        size = len(cluster_scores)
        y_upper = y_lower + size
        
        color = cm.nipy_spectral(float(i) / len(unique_labels))
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_scores,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label with task name
        original_label = unique_labels[i]
        task_name = task_names_mapping[original_label]
        plt.text(-0.05, y_lower + 0.5 * size, task_name)
        
        y_lower = y_upper + 10
    
    plt.axvline(x=avg_score, color="red", linestyle="--",
                label=f"Average Score: {avg_score:.4f}")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Task Clusters")
    plt.title("Silhouette Analysis - Task Clustering Quality")
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "silhouette_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Silhouette analysis saved to {output_dir}/silhouette_analysis.png")
    return avg_score


def create_tsne_visualization(embeddings_dict, output_dir):
    """
    Create t-SNE visualization of model embeddings.
    
    This provides a 2D visualization of how models cluster by task.
    """
    if len(embeddings_dict) <= 1:
        logger.warning("Need at least 2 models for t-SNE")
        return
    
    logger.info("Creating t-SNE visualization...")
    
    model_names = list(embeddings_dict.keys())
    embeddings_matrix = np.stack([embeddings_dict[name] for name in model_names])
    
    # Apply t-SNE with perplexity=2 as requested
    n_models = len(model_names)
    perplexity = min(2, n_models - 1)  # Ensure valid perplexity
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_matrix)
    
    # Get task categories and colors
    labels, task_names_list = categorize_models(model_names)
    
    # Define colors for each task
    colors_map = {
        "LegalBench": "#1f77b4",  # Blue
        "GSM8K": "#ff7f0e",       # Orange  
        "PubMedQA": "#2ca02c",    # Green
        "HellaSwag": "#d62728",   # Red
        "OPC-SFT": "#9467bd",     # Purple
        "Unknown": "#7f7f7f"      # Gray
    }
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each point with appropriate color
    for i, (x, y) in enumerate(embeddings_2d):
        task_name = task_names_list[i]
        color = colors_map.get(task_name, colors_map["Unknown"])
        plt.scatter(x, y, color=color, s=100, alpha=0.7)
    
    # Create legend
    legend_elements = []
    unique_tasks = list(set(task_names_list))
    for task in sorted(unique_tasks):
        if task != "Unknown":
            color = colors_map.get(task, colors_map["Unknown"])
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      markersize=10, label=task)
            )
    
    plt.legend(handles=legend_elements, title="Tasks", loc='best')
    plt.title(f't-SNE Visualization (perplexity={perplexity})', fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "tsne_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"t-SNE visualization saved to {output_dir}/tsne_visualization.png")


def main():
    """Main analysis function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model configuration
    logger.info(f"Analyzing {args.model} model family...")
    config = get_model_config(args.model)
    base_model_name = config["base_model"]
    model_paths = config["models"]
    
    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Analyzing {len(model_paths)} fine-tuned models")
    
    # Log initial memory
    log_memory_usage()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cuda")
    base_model.config.output_hidden_states = True
    base_model.eval()
    
    log_memory_usage()
    
    # Create generic probes
    probe_texts = create_generic_probes()
    
    # Get base model activation
    logger.info("Computing base model activations...")
    base_activation = get_average_activation(base_model, probe_texts, tokenizer, device)
    
    # Process each fine-tuned model
    logger.info("Processing fine-tuned models...")
    embeddings = {}
    
    for i, model_path in enumerate(model_paths):
        logger.info(f"Processing model {i+1}/{len(model_paths)}: {model_path}")
        
        try:
            with load_model_temporarily(model_path, base_model, tokenizer, device) as model:
                # Compute fine-tuned model activation
                finetuned_activation = get_average_activation(model, probe_texts, tokenizer, device)
                
                # Compute delta (fine-tuned - base)
                delta_embedding = finetuned_activation - base_activation
                embeddings[model_path] = delta_embedding
                
                logger.info(f"Successfully processed {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to process {model_path}: {e}")
            continue
        
        log_memory_usage()
    
    # Clean up base model
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    log_memory_usage()
    
    logger.info(f"Successfully processed {len(embeddings)}/{len(model_paths)} models")
    
    if len(embeddings) < 3:
        logger.error("Need at least 3 models for analysis. Exiting.")
        return
    
    # Run analysis
    logger.info("Running clustering analysis...")
    
    # Compute silhouette score
    silhouette_score = compute_silhouette_analysis(embeddings, args.output_dir)
    
    # Create t-SNE visualization  
    create_tsne_visualization(embeddings, args.output_dir)
    
    # Save results summary
    results = {
        "model_family": args.model,
        "base_model": base_model_name,
        "num_models_analyzed": len(embeddings),
        "silhouette_score": float(silhouette_score) if silhouette_score is not None else None,
        "model_paths": list(embeddings.keys())
    }
    
    with open(os.path.join(args.output_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Model Family: {args.model}")
    print(f"Models Analyzed: {len(embeddings)}")
    if silhouette_score is not None:
        print(f"Silhouette Score: {silhouette_score:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()