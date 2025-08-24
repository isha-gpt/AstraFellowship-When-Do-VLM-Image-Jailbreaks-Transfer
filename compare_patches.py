import torch
import numpy as np

def load_and_compare_patches():
    # Load the tensors
    patch_3500 = torch.load("projected_patch_step=3500.pt")
    patch_10500 = torch.load("projected_patch_step=10500.pt")
    
    # Basic shape comparison
    print(f"Shape of patch_3500: {patch_3500.shape}")
    print(f"Shape of patch_10500: {patch_10500.shape}")
    
    # Check if tensors are exactly equal
    are_equal = torch.equal(patch_3500, patch_10500)
    print(f"\nAre tensors exactly equal? {are_equal}")
    
    # If not exactly equal, compute some statistics
    if not are_equal:
        # Compute element-wise difference
        diff = patch_3500 - patch_10500
        print(f"\nMaximum absolute difference: {diff.abs().max().item()}")
        print(f"Mean absolute difference: {diff.abs().mean().item()}")
        print(f"Standard deviation of differences: {diff.std().item()}")
        
        # Compare norms
        print(f"\nNorm of patch_3500: {patch_3500.norm().item()}")
        print(f"Norm of patch_10500: {patch_10500.norm().item()}")
        
        # Compare first few elements
        print("\nFirst few elements of patch_3500:")
        print(patch_3500[0][0][:5])
        print("\nFirst few elements of patch_10500:")
        print(patch_10500[0][0][:5])

if __name__ == "__main__":
    load_and_compare_patches() 