#!/usr/bin/env python

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import argparse
import pickle
from pathlib import Path
from torchvision import transforms
from datasets import load_dataset
from prismatic import load
from src.globals import PRISM_IDS, CACHE_DIR, HF_CACHE_DIR, DATASET_PATHS

# Set up environment
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Get HF token from environment or prompt
try:
    from dotenv import load_dotenv
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        hf_token = input("Enter your HuggingFace token: ")
except ImportError:
    hf_token = input("Enter your HuggingFace token: ")

# Define all models and encoders
MODELS = [
    "llama3",  # Llama-3-8B-Instruct
    "llama2",  # Llama-2-7b-chat-hf
    "mistralv1",  # Mistral-7B-Instruct-v0.1
    "mistralv2"   # Mistral-7B-Instruct-v0.2
]

ENCODERS = [
    "clip",
    "dinosiglip",
    "siglip"
]

def get_model_key(model, encoder):
    """Get the model key for PRISM_IDS dictionary."""
    return f"{model}+{encoder}"

def get_model_name(model, encoder):
    """Get a display name for the model."""
    model_display = {
        "llama3": "Llama-3-8B",
        "llama2": "Llama-2-7B",
        "mistralv1": "Mistral-v0.1",
        "mistralv2": "Mistral-v0.2"
    }
    return f"{model_display[model]}\n+{encoder}"

def get_cache_path(model_key, location, num_images):
    """Get the cache file path for a specific model and configuration."""
    cache_dir = Path("results/cached_representations")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model_key}_{location}_{num_images}.pkl"

def load_cached_representations(model_key, location, num_images):
    """Load cached representations if they exist."""
    cache_path = get_cache_path(model_key, location, num_images)
    if cache_path.exists():
        print(f"Loading cached representations for {model_key}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_cached_representations(model_key, location, num_images, representations):
    """Save representations to cache."""
    cache_path = get_cache_path(model_key, location, num_images)
    print(f"Saving representations for {model_key} to cache...")
    with open(cache_path, 'wb') as f:
        pickle.dump(representations, f)

def get_vision_encoder_representations(model, images, device="cuda:0"):
    """
    Get the raw representations from the vision encoder output (before projection).
    
    Returns: A tensor of shape [num_images, vision_hidden_size]
    """
    all_representations = []
    
    for image in tqdm(images, desc=f"Processing images through vision encoder"):
        try:
            with torch.no_grad():
                # Process the image through the vision transform
                pixel_values = model.vision_backbone.image_transform(image)
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.unsqueeze(0).to(device)
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: v.unsqueeze(0).to(device) for k, v in pixel_values.items()}
                
                # Get vision encoder outputs - these are the patch features directly
                vision_features = model.vision_backbone(pixel_values)
                
                # If the features are a tensor, use them directly
                # If they're a dict, we need to handle it appropriately
                if isinstance(vision_features, torch.Tensor):
                    # Take mean of patch features to get a single vector per image
                    features = vision_features.mean(dim=1)  # [1, vision_hidden_size]
                elif isinstance(vision_features, dict):
                    # Handle dictionary case - take mean of the main feature tensor
                    features = vision_features['last_hidden_state'].mean(dim=1)  # [1, vision_hidden_size]
                else:
                    raise ValueError(f"Unexpected vision feature type: {type(vision_features)}")
                
                # Don't project - use raw vision encoder features
                all_representations.append(features.cpu())
        except Exception as e:
            print(f"Error processing image: {e}")
            if all_representations:
                all_representations.append(torch.zeros_like(all_representations[-1]))
            else:
                print("Skipping this image due to error")
                continue
    
    if all_representations:
        return torch.cat(all_representations, dim=0)  # [num_images, vision_hidden_size]
    else:
        raise ValueError("No images could be processed successfully")

def get_post_projector_representations(model, images, device="cuda:0"):
    """
    Get the representations after the projector but before the language model.
    
    Returns: A tensor of shape [num_images, projected_size]
    """
    all_representations = []
    
    for image in tqdm(images, desc=f"Processing images through projector"):
        try:
            with torch.no_grad():
                # Process the image through the vision transform
                pixel_values = model.vision_backbone.image_transform(image)
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.unsqueeze(0).to(device)
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: v.unsqueeze(0).to(device) for k, v in pixel_values.items()}
                
                # Get vision encoder outputs
                vision_features = model.vision_backbone(pixel_values)
                
                # Handle different feature formats
                if isinstance(vision_features, torch.Tensor):
                    features = vision_features.mean(dim=1)  # [1, hidden_size]
                elif isinstance(vision_features, dict):
                    features = vision_features['last_hidden_state'].mean(dim=1)  # [1, hidden_size]
                else:
                    raise ValueError(f"Unexpected vision feature type: {type(vision_features)}")
                
                # Project the features through the model's projector
                projected_features = model.projector(features)  # [1, projected_size]
                
                all_representations.append(projected_features.cpu())
        except Exception as e:
            print(f"Error processing image: {e}")
            if all_representations:
                all_representations.append(torch.zeros_like(all_representations[-1]))
            else:
                print("Skipping this image due to error")
                continue
    
    if all_representations:
        return torch.cat(all_representations, dim=0)  # [num_images, projected_size]
    else:
        raise ValueError("No images could be processed successfully")

def get_llm_final_layer_representations(model, images, device="cuda:0"):
    """
    Get the CLS token representation (first token) of the language model when processing images with an empty prompt.
    
    Returns: A tensor of shape [num_images, hidden_size]
    """
    all_representations = []
    
    for image in tqdm(images, desc=f"Processing images through LLM"):
        try:
            with torch.no_grad():
                # Create an empty prompt
                prompt_builder = model.get_prompt_builder()
                prompt_builder.add_turn(role="human", message="")
                empty_prompt = prompt_builder.get_prompt()
                
                # Tokenize the empty prompt
                tokenizer = model.llm_backbone.tokenizer
                input_ids = tokenizer(empty_prompt, return_tensors="pt").input_ids.to(device)
                
                # Process the image through the vision transform
                pixel_values = model.vision_backbone.image_transform(image)
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.unsqueeze(0).to(device)
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: v.unsqueeze(0).to(device) for k, v in pixel_values.items()}
                
                # Pass both input_ids and pixel_values to the model's forward function
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                )
                
                # Get the final layer hidden state - CLS token (first token)
                last_hidden_state = outputs.hidden_states[-1]  # Last layer
                cls_token_hidden_state = last_hidden_state[0, 0, :]  # First token (CLS)
                
                all_representations.append(cls_token_hidden_state.cpu())
        except Exception as e:
            print(f"Error processing image: {e}")
            # If an image fails, append a tensor of zeros to maintain ordering
            if all_representations:
                # Use the same size as the previous representations
                all_representations.append(torch.zeros_like(all_representations[-1]))
            else:
                # If this is the first image, we can't know the size, so skip
                print("Skipping this image due to error")
                continue
    
    # Stack all representations (if any were successful)
    if all_representations:
        return torch.stack(all_representations)
    else:
        raise ValueError("No images could be processed successfully")

def compute_cosine_similarity_matrix(models_dict):
    """
    Compute the cosine similarity matrix between all model-encoder combinations.
    Returns: A dictionary with keys 'similarity_matrix' and 'labels'
    """
    model_keys = list(models_dict.keys())
    n = len(model_keys)
    similarity_matrix = np.zeros((n, n))
    
    # Print dimensions for debugging
    print("\nRepresentation dimensions:")
    for model_key, reps in models_dict.items():
        print(f"  {model_key}: {reps.shape}")
    
    for i, (model_key_i, reps_i) in enumerate(models_dict.items()):
        for j, (model_key_j, reps_j) in enumerate(models_dict.items()):
            # Check if dimensions are compatible for cosine similarity
            if reps_i.shape[1] != reps_j.shape[1]:
                print(f"Warning: Incompatible dimensions for {model_key_i} ({reps_i.shape[1]}) and {model_key_j} ({reps_j.shape[1]})")
                similarity_matrix[i, j] = np.nan  # Use NaN for incompatible dimensions
                continue
            
            try:
                # Normalize the representations
                reps_i_norm = reps_i / reps_i.norm(dim=1, keepdim=True)
                reps_j_norm = reps_j / reps_j.norm(dim=1, keepdim=True)
                
                # Compute cosine similarity for each pair
                similarities = torch.matmul(reps_i_norm, reps_j_norm.t())
                
                # Take the average over the diagonal (same input similarity)
                avg_similarity = torch.mean(torch.diag(similarities)).item()
                similarity_matrix[i, j] = avg_similarity
            except Exception as e:
                print(f"Error computing similarity between {model_key_i} and {model_key_j}: {e}")
                similarity_matrix[i, j] = np.nan
    
    # Create formatted labels
    labels = []
    for model_key in model_keys:
        parts = model_key.split('+')
        if len(parts) == 2:
            model, encoder = parts
            if model in {"llama3", "llama2", "mistralv1", "mistralv2"}:
                model_display = {
                    "llama3": "Llama-3-8B",
                    "llama2": "Llama-2-7B",
                    "mistralv1": "Mistral-v0.1",
                    "mistralv2": "Mistral-v0.2"
                }
                labels.append(f"{model_display[model]}\n+{encoder}")
            else:
                labels.append(model_key)
    
    return {
        'similarity_matrix': similarity_matrix,
        'labels': labels
    }

def compute_centered_kernel_alignment_matrix(models_dict):
    """
    Compute the Centered Kernel Alignment (CKA) matrix between all model-encoder combinations.
    Returns: A dictionary with keys 'similarity_matrix' and 'labels'
    """
    model_keys = list(models_dict.keys())
    n = len(model_keys)
    cka_matrix = np.zeros((n, n))
    
    # Print dimensions for debugging
    print("\nRepresentation dimensions:")
    for model_key, reps in models_dict.items():
        print(f"  {model_key}: {reps.shape}")
    
    # Print sample statistics and first row for each model
    print('Sample representations for each model:')
    for model_key, reps in models_dict.items():
        print(f'{model_key}: mean={reps.mean().item()}, std={reps.std().item()}, min={reps.min().item()}, max={reps.max().item()}')
        print(f'  First row: {reps[0][:10]}')  # print first 10 values of first image
    
    def cka(X, Y):
        """Compute Centered Kernel Alignment between X and Y."""
        # Center the data
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute Gram matrices
        K = torch.matmul(X_centered, X_centered.t())
        L = torch.matmul(Y_centered, Y_centered.t())
        
        # Compute CKA
        numerator = torch.trace(torch.matmul(K, L))
        denominator = torch.sqrt(torch.trace(torch.matmul(K, K)) * torch.trace(torch.matmul(L, L)))
        
        return (numerator / denominator).item()
    
    for i, (model_key_i, reps_i) in enumerate(models_dict.items()):
        for j, (model_key_j, reps_j) in enumerate(models_dict.items()):
            # Check if dimensions are compatible for CKA
            if reps_i.shape[1] != reps_j.shape[1]:
                print(f"Warning: Incompatible dimensions for {model_key_i} ({reps_i.shape[1]}) and {model_key_j} ({reps_j.shape[1]})")
                cka_matrix[i, j] = np.nan  # Use NaN for incompatible dimensions
                continue
            try:
                cka_value = cka(reps_i, reps_j)
                cka_matrix[i, j] = cka_value
            except Exception as e:
                print(f"Error computing CKA between {model_key_i} and {model_key_j}: {e}")
                cka_matrix[i, j] = np.nan
    
    # Create formatted labels
    labels = []
    for model_key in model_keys:
        parts = model_key.split('+')
        if len(parts) == 2:
            model, encoder = parts
            if model in {"llama3", "llama2", "mistralv1", "mistralv2"}:
                model_display = {
                    "llama3": "Llama-3-8B",
                    "llama2": "Llama-2-7B",
                    "mistralv1": "Mistral-v0.1",
                    "mistralv2": "Mistral-v0.2"
                }
                labels.append(f"{model_display[model]}\n+{encoder}")
            else:
                labels.append(model_key)
    
    return {
        'similarity_matrix': cka_matrix,
        'labels': labels
    }

def plot_heatmap(similarity_data, title, filename, metric_name="Similarity"):
    """Create and save a heatmap from the similarity matrix."""
    plt.figure(figsize=(14, 12))
    
    # Determine color range based on metric
    if metric_name.lower() == "cka":
        vmin, vmax = 0, 1
        cmap = "YlOrRd"
    else:  # cosine similarity
        vmin, vmax = 0, 1
        cmap = "YlOrRd"
    
    # Create a mask for NaN values
    mask = np.isnan(similarity_data['similarity_matrix'])
    
    # Create the heatmap
    sns.heatmap(
        similarity_data['similarity_matrix'],
        annot=True,
        fmt=".3f",
        cmap=cmap,
        xticklabels=similarity_data['labels'],
        yticklabels=similarity_data['labels'],
        vmin=vmin, vmax=vmax,
        mask=mask,
        cbar_kws={'label': f'{metric_name.upper()} Score'}
    )
    
    # Add a note about NaN values if any exist
    if np.any(mask):
        plt.figtext(0.5, 0.02, "White cells indicate incompatible dimensions (NaN values)", 
                   ha='center', fontsize=10, style='italic')
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {filename}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare VLM representations at different stages.')
    parser.add_argument('--location', type=str, 
                        choices=['vision_encoder', 'post_projector', 'llm_final_layer'], 
                        default='llm_final_layer',
                        help='Location in the VLM to extract representations from')
    parser.add_argument('--num_images', type=int, default=50,
                        help='Number of images from CIFAR-10 to use')
    parser.add_argument('--metric', type=str, 
                        choices=['cosine', 'cka'], 
                        default='cosine',
                        help='Similarity metric to use: cosine similarity or centered kernel alignment (CKA)')
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force recomputation of representations even if cached versions exist')
    args = parser.parse_args()
    
    # Create a directory for results if it doesn't exist
    os.makedirs("results/heatmaps", exist_ok=True)
    
    # Load CIFAR-10 dataset
    print(f"Loading CIFAR-10 dataset ({args.num_images} images)...")
    dataset = load_dataset("cifar10", split=f"test[:{args.num_images}]")
    images = []
    for item in tqdm(dataset, desc="Preparing images"):
        images.append(item["img"])
    num_inputs = len(images)
    print(f"Loaded {num_inputs} images from CIFAR-10")
    
    # Map location to function
    location_functions = {
        'vision_encoder': get_vision_encoder_representations,
        'post_projector': get_post_projector_representations,
        'llm_final_layer': get_llm_final_layer_representations
    }
    
    get_representations_func = location_functions[args.location]
    
    # Map metric to function
    metric_functions = {
        'cosine': compute_cosine_similarity_matrix,
        'cka': compute_centered_kernel_alignment_matrix
    }
    
    compute_similarity_func = metric_functions[args.metric]
    
    # Collect representations for all model-encoder combinations
    all_combinations = {}
    
    print(f"Loading models and computing representations for {len(MODELS)} models with {len(ENCODERS)} encoders...")
    for model in MODELS:
        for encoder in ENCODERS:
            model_key = get_model_key(model, encoder)
            if model_key in PRISM_IDS:
                try:
                    # Check if cached representations exist
                    cached_reps = None
                    if not args.force_recompute:
                        cached_reps = load_cached_representations(model_key, args.location, args.num_images)
                    
                    if cached_reps is not None:
                        print(f"Using cached representations for {model_key}")
                        all_combinations[model_key] = cached_reps
                    else:
                        print(f"Loading {model_key} and computing representations...")
                        # Load model
                        prismatic_model, tokenizer = load(PRISM_IDS[model_key], hf_token=hf_token, cache_dir=CACHE_DIR)
                        prismatic_model = prismatic_model.to("cuda:0")
                        
                        # Get representations based on location
                        representations = get_representations_func(prismatic_model, images)
                        all_combinations[model_key] = representations
                        
                        # Save to cache
                        save_cached_representations(model_key, args.location, args.num_images, representations)
                        
                        # Clean up to save memory
                        del prismatic_model
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error loading or processing {model_key}: {e}")
            else:
                print(f"Model key {model_key} not found in PRISM_IDS")
    
    # Compute similarity matrix for all combinations
    metric_display = "Cosine Similarity" if args.metric == 'cosine' else "Centered Kernel Alignment (CKA)"
    location_display = {
        'vision_encoder': 'Post Vision Encoder',
        'post_projector': 'Post Projector',
        'llm_final_layer': 'LLM Final Layer (CLS Token)'
    }[args.location]
    
    print(f"Computing {args.metric.upper()} matrix for {len(all_combinations)} model-encoder combinations...")
    similarity_data = compute_similarity_func(all_combinations)
    
    # Plot the heatmap for all combinations
    title = f"Average {metric_display} Between {location_display} Representations\n(Different Models with Different Vision Encoders on CIFAR-10, {num_inputs} images)"
    filename = f"results/heatmaps/cifar10_{args.location}_{args.metric}.png"
    plot_heatmap(similarity_data, title, filename, args.metric)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 
