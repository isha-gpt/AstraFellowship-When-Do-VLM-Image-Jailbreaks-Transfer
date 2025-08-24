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
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import src.globals

# Set up environment
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Get HF token from environment
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    # Try loading from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
    except ImportError:
        pass
    
    if not hf_token:
        hf_token = input("Enter your HuggingFace token: ")

# Define cache directory
CACHE_DIR = os.path.expanduser("~/.cache/prismatic")
os.makedirs(CACHE_DIR, exist_ok=True)

# Define the subset of models we'll use based on what's in the transfer data
MODELS_TO_USE = [
    "prism-llama3-instruct+8b+clip",
    "prism-llama3-instruct+8b+siglip", 
    "prism-llama3-instruct+8b+dinosiglip",
    "prism-llama2-chat+7b+clip",
    "prism-llama2-chat+7b+siglip",
    "prism-llama2-chat+7b+dinosiglip",
    "prism-mistral-instruct-v0.2+7b+clip",
    "prism-mistral-instruct-v0.2+7b+siglip",
    "prism-mistral-instruct-v0.2+7b+dinosiglip",
]

# Map from prism model IDs to simple keys for caching
PRISM_TO_SIMPLE_KEY = {
    "prism-llama3-instruct+8b+clip": "llama3+clip",
    "prism-llama3-instruct+8b+siglip": "llama3+siglip",
    "prism-llama3-instruct+8b+dinosiglip": "llama3+dinosiglip",
    "prism-llama2-chat+7b+clip": "llama2+clip",
    "prism-llama2-chat+7b+siglip": "llama2+siglip",
    "prism-llama2-chat+7b+dinosiglip": "llama2+dinosiglip",
    "prism-mistral-instruct-v0.2+7b+clip": "mistralv2+clip",
    "prism-mistral-instruct-v0.2+7b+siglip": "mistralv2+siglip",
    "prism-mistral-instruct-v0.2+7b+dinosiglip": "mistralv2+dinosiglip",
}

# Use the prism- prefixed IDs which will trigger download from RylanSchaeffer/prismatic-vlms
PRISM_IDS = {
    "prism-llama3-instruct+8b+clip": "prism-llama3-instruct+8b+clip",
    "prism-llama3-instruct+8b+siglip": "prism-llama3-instruct+8b+siglip", 
    "prism-llama3-instruct+8b+dinosiglip": "prism-llama3-instruct+8b+dinosiglip",
    "prism-llama2-chat+7b+clip": "prism-llama2-chat+7b+clip",
    "prism-llama2-chat+7b+siglip": "prism-llama2-chat+7b+siglip",
    "prism-llama2-chat+7b+dinosiglip": "prism-llama2-chat+7b+dinosiglip",
    "prism-mistral-instruct-v0.2+7b+clip": "prism-mistral-instruct-v0.2+7b+clip",
    "prism-mistral-instruct-v0.2+7b+siglip": "prism-mistral-instruct-v0.2+7b+siglip",
    "prism-mistral-instruct-v0.2+7b+dinosiglip": "prism-mistral-instruct-v0.2+7b+dinosiglip",
}

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

def get_llm_final_layer_representations(model, images, device="cuda:1"):
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

def get_vision_encoder_representations(model, images, device="cuda:1"):
    """
    Get the vision encoder representations (before projection to language space).
    
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
                
                # Get vision encoder outputs
                vision_outputs = model.vision_backbone(pixel_values, output_hidden_states=True)
                
                # Get the final vision encoder representation (usually the last hidden state)
                if hasattr(vision_outputs, 'last_hidden_state'):
                    vision_repr = vision_outputs.last_hidden_state
                elif hasattr(vision_outputs, 'hidden_states'):
                    vision_repr = vision_outputs.hidden_states[-1]
                else:
                    # Fallback: try to get the pooled output
                    vision_repr = vision_outputs.pooler_output.unsqueeze(1)
                
                # Take the CLS token or mean pooling
                if vision_repr.shape[1] > 1:  # Multiple tokens
                    # Use CLS token (first token) or mean pooling
                    if hasattr(vision_outputs, 'pooler_output'):
                        vision_repr = vision_outputs.pooler_output
                    else:
                        vision_repr = vision_repr[:, 0, :]  # CLS token
                else:
                    vision_repr = vision_repr.squeeze(1)
                
                all_representations.append(vision_repr.cpu())
        except Exception as e:
            print(f"Error processing image through vision encoder: {e}")
            if all_representations:
                all_representations.append(torch.zeros_like(all_representations[-1]))
            else:
                print("Skipping this image due to error")
                continue
    
    if all_representations:
        return torch.stack(all_representations)
    else:
        raise ValueError("No images could be processed successfully")

def get_post_projector_representations(model, images, device="cuda:1"):
    """
    Get the representations after the vision-to-language projection layer.
    
    Returns: A tensor of shape [num_images, projected_hidden_size]
    """
    all_representations = []
    
    for image in tqdm(images, desc=f"Processing images through post-projector"):
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
                
                # Get vision encoder outputs first
                vision_outputs = model.vision_backbone(pixel_values, output_hidden_states=True)
                
                # Get the final vision encoder representation
                if hasattr(vision_outputs, 'last_hidden_state'):
                    vision_repr = vision_outputs.last_hidden_state
                elif hasattr(vision_outputs, 'hidden_states'):
                    vision_repr = vision_outputs.hidden_states[-1]
                else:
                    vision_repr = vision_outputs.pooler_output.unsqueeze(1)
                
                # Take CLS token or mean pooling
                if vision_repr.shape[1] > 1:
                    if hasattr(vision_outputs, 'pooler_output'):
                        vision_repr = vision_outputs.pooler_output
                    else:
                        vision_repr = vision_repr[:, 0, :]
                else:
                    vision_repr = vision_repr.squeeze(1)
                
                # Apply the projection layer to convert vision to language space
                if hasattr(model, 'vision_to_language_projection'):
                    projected_repr = model.vision_to_language_projection(vision_repr)
                elif hasattr(model, 'projector'):
                    projected_repr = model.projector(vision_repr)
                else:
                    # Try to find the projection layer in the model
                    for name, module in model.named_modules():
                        if 'projection' in name.lower() or 'projector' in name.lower():
                            projected_repr = module(vision_repr)
                            break
                    else:
                        # If no projection layer found, use vision repr as is
                        projected_repr = vision_repr
                
                all_representations.append(projected_repr.cpu())
        except Exception as e:
            print(f"Error processing image through post-projector: {e}")
            if all_representations:
                all_representations.append(torch.zeros_like(all_representations[-1]))
            else:
                print("Skipping this image due to error")
                continue
    
    if all_representations:
        return torch.stack(all_representations)
    else:
        raise ValueError("No images could be processed successfully")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate cached representations for VLM comparison.')
    parser.add_argument('--location', type=str, 
                        choices=['vision_encoder', 'post_projector', 'llm_final_layer'], 
                        default='llm_final_layer',
                        help='Location in the VLM to extract representations from')
    parser.add_argument('--num_images', type=int, default=50,
                        help='Number of images from CIFAR-10 to use')
    parser.add_argument('--metric', type=str, 
                        choices=['cosine', 'cka'], 
                        default='cosine',
                        help='Similarity metric to use (for compatibility with original script)')
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force recomputation of representations even if cached versions exist')
    args = parser.parse_args()
    
    # Create a directory for results if it doesn't exist
    os.makedirs("results/cached_representations", exist_ok=True)
    
    # Load CIFAR-10 dataset
    print(f"Loading CIFAR-10 dataset ({args.num_images} images)...")
    dataset = load_dataset("cifar10", split=f"test[:{args.num_images}]")
    images = []
    for item in tqdm(dataset, desc="Preparing images"):
        images.append(item["img"])
    num_inputs = len(images)
    print(f"Loaded {num_inputs} images from CIFAR-10")
    
    # Map location to the appropriate function
    location_to_func = {
        'vision_encoder': get_vision_encoder_representations,
        'post_projector': get_post_projector_representations,
        'llm_final_layer': get_llm_final_layer_representations,
    }
    
    get_representations_func = location_to_func[args.location]
    
    # Generate representations for each model
    success_count = 0
    for prism_model_key in MODELS_TO_USE:
        simple_key = PRISM_TO_SIMPLE_KEY[prism_model_key]
        
        try:
            # Check if cached representations exist
            cached_reps = None
            if not args.force_recompute:
                cached_reps = load_cached_representations(simple_key, args.location, args.num_images)
            
            if cached_reps is not None:
                print(f"Using cached representations for {simple_key}")
                success_count += 1
            else:
                print(f"Loading {prism_model_key} and computing representations...")
                
                # Check if we have the model ID
                if prism_model_key not in PRISM_IDS:
                    print(f"Warning: No HF model ID found for {prism_model_key}, skipping...")
                    continue
                
                try:
                    # Clear GPU cache before loading
                    import torch
                    torch.cuda.empty_cache()
                    
                    # Load model (prismatic load() returns only the model, not a tuple)
                    print(f"Loading model {prism_model_key}...")
                    prismatic_model = load(PRISM_IDS[prism_model_key], hf_token=hf_token, cache_dir=CACHE_DIR)
                    
                    print(f"Moving model to GPU...")
                    prismatic_model = prismatic_model.to("cuda:1")
                    print(f"Model loaded and moved to GPU successfully")
                    
                    # Get representations
                    representations = get_representations_func(prismatic_model, images)
                    
                    # Save to cache using simple key
                    save_cached_representations(simple_key, args.location, args.num_images, representations)
                    success_count += 1
                    
                    # Clean up to save memory
                    print(f"Cleaning up model {prism_model_key}...")
                    del prismatic_model
                    torch.cuda.empty_cache()
                    print(f"Memory cleaned up")
                    
                except Exception as e:
                    print(f"Error loading or processing {prism_model_key}: {e}")
                    print("This might be due to missing HF model or authentication issues.")
                    
        except Exception as e:
            print(f"Error with {prism_model_key}: {e}")
    
    print(f"\nSuccessfully processed {success_count}/{len(MODELS_TO_USE)} models")
    if success_count > 0:
        print("Cached representations are now available for the combine_similarity_transfer.py script!")
    else:
        print("No representations were successfully generated. Please check your setup and HF token.")

if __name__ == "__main__":
    main() 