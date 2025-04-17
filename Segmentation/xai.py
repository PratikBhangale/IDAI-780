# saliency_visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from segmentation import BrainMRIDataset, load_and_preprocess_data, train_val_test_split
from torch.utils.data import DataLoader
import os

def load_model(model_path, device='cuda'):
    """Load a trained UNet model."""
    model = torch.jit.load(model_path)
    model.to(device)
    model.eval()
    return model

def compute_saliency_map(model, image, target_class=None, threshold=0.5):
    """
    Compute the saliency map for the input image.
    
    Args:
        model: Trained UNet model
        image: Input image tensor (1, 1, H, W)
        target_class: Not used for segmentation, but kept for API consistency
        threshold: Threshold for considering positive predictions
        
    Returns:
        saliency_map: Gradient of output with respect to input
    """
    # Clone the image and enable gradient tracking
    image = image.clone().detach().requires_grad_(True)
    
    # Get model predictions
    output = model(image)
    
    # For segmentation, we care about the gradient of all positive predictions
    # Create a mask of where the model predicts positively
    mask = (output > threshold).float()
    
    # If mask is empty (no positive predictions), use the raw output
    if mask.sum() == 0:
        mask = output
    
    # Backpropagate to compute gradients
    model.zero_grad()
    output.backward(gradient=mask)
    
    # Get gradients with respect to input
    saliency_map = image.grad.abs()
    
    return saliency_map

def normalize_saliency_map(saliency_map):
    """Normalize saliency map for visualization."""
    saliency_map = saliency_map.detach().cpu().numpy()[0, 0]
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    return saliency_map

def overlay_saliency_map(image, saliency_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay saliency map on original image."""
    # Convert to numpy and scale to [0, 255]
    image_np = image.detach().cpu().numpy()[0, 0] * 255
    saliency_np = saliency_map * 255
    
    # Convert to RGB for visualization
    image_rgb = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(saliency_np.astype(np.uint8), colormap)
    
    # Overlay
    overlay = cv2.addWeighted(image_rgb, 1-alpha, heatmap, alpha, 0)
    
    return overlay

def visualize_saliency(image, mask, saliency_map, prediction, save_path=None, 
                       figsize=(20, 5), show_plot=True, prediction_threshold=0.5):
    """Visualize input, ground truth, prediction and saliency map."""
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original image
    axes[0].imshow(image.detach().cpu().numpy()[0, 0], cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask.detach().cpu().numpy()[0, 0], cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction.detach().cpu().numpy()[0, 0] > prediction_threshold, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Saliency map
    normalized_saliency = normalize_saliency_map(saliency_map)
    axes[3].imshow(normalized_saliency, cmap='hot')
    axes[3].set_title('Saliency Map')
    axes[3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_without_groundtruth(image, prediction, saliency_map, save_path=None, 
                                 figsize=(15, 5), show_plot=True, prediction_threshold=0.5):
    """Visualize input, prediction and saliency map without ground truth."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(image.detach().cpu().numpy()[0, 0], cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(prediction.detach().cpu().numpy()[0, 0] > prediction_threshold, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Saliency map
    normalized_saliency = normalize_saliency_map(saliency_map)
    axes[2].imshow(normalized_saliency, cmap='hot')
    axes[2].set_title('Saliency Map')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def preprocess_external_image(image_path, size=(256, 256)):
    """Preprocess an external image to match the model's expected input."""
    # Load and convert to grayscale if needed
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize to model's expected input size
    img = cv2.resize(img, size)
    # Normalize
    img = img / 255.0
    # Add batch and channel dimensions
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    return img

def run_saliency_analysis(
    model_path='best_Base_Unet_scripted.pt',
    data_dir='kaggle_3m',
    output_dir='saliency_maps',
    num_samples=5,
    device=None,
    prediction_threshold=0.5,
    saliency_threshold=0.5,
    overlay_alpha=0.5,
    colormap=cv2.COLORMAP_JET,
    show_plots=True,
    save_plots=True,
    external_image_path=None,
    sample_indices=None
):
    """
    Run saliency map analysis with configurable parameters.
    
    Args:
        model_path: Path to the trained model
        data_dir: Directory containing the dataset
        output_dir: Directory to save output visualizations
        num_samples: Number of test samples to process
        device: Device to run on ('cuda' or 'cpu')
        prediction_threshold: Threshold for binary segmentation
        saliency_threshold: Threshold for saliency map computation
        overlay_alpha: Alpha value for saliency map overlay
        colormap: Colormap for saliency visualization
        show_plots: Whether to display the plots
        save_plots: Whether to save the plots
        external_image_path: Path to an external image to analyze
        sample_indices: Specific indices of test samples to analyze
    """
    # Device setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Create output directory if needed
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process external image if provided
    if external_image_path:
        try:
            # Process external image
            external_image = preprocess_external_image(external_image_path).to(device)
            
            # Get prediction
            with torch.no_grad():
                external_prediction = model(external_image)
            
            # Compute saliency map
            external_saliency = compute_saliency_map(model, external_image, threshold=saliency_threshold)
            
            # Visualize
            ext_save_path = os.path.join(output_dir, "external_image_saliency.png") if save_plots else None
            visualize_without_groundtruth(
                external_image, 
                external_prediction, 
                external_saliency,
                save_path=ext_save_path,
                show_plot=show_plots,
                prediction_threshold=prediction_threshold
            )
            
            # Save overlay
            if save_plots:
                normalized_saliency = normalize_saliency_map(external_saliency)
                overlay = overlay_saliency_map(
                    external_image, 
                    normalized_saliency, 
                    alpha=overlay_alpha,
                    colormap=colormap
                )
                
                overlay_path = os.path.join(output_dir, "external_image_overlay.png")
                plt.figure(figsize=(8, 8))
                plt.imshow(overlay)
                plt.title('Saliency Overlay')
                plt.axis('off')
                plt.savefig(overlay_path)
                if not show_plots:
                    plt.close()
                
                print(f"Processed external image, saved to {ext_save_path} and {overlay_path}")
            
        except Exception as e:
            print(f"Error processing external image: {e}")
    
    # Process dataset samples
    if data_dir:
        # Load data
        print("Loading data...")
        images, masks = load_and_preprocess_data(data_dir)
        _, _, (test_images, test_masks) = train_val_test_split(images, masks)
        
        # Create test dataset
        test_dataset = BrainMRIDataset(test_images, test_masks, augment=False)
        
        # If specific indices are provided, use them
        if sample_indices:
            indices_to_process = sample_indices
        else:
            # Otherwise use the first num_samples
            indices_to_process = range(min(num_samples, len(test_dataset)))
        
        # Process each sample
        for idx in indices_to_process:
            if idx >= len(test_dataset):
                print(f"Warning: Index {idx} out of range. Dataset has {len(test_dataset)} samples.")
                continue
                
            image, mask = test_dataset[idx]
            # Add batch dimension
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                prediction = model(image)
            
            # Compute saliency map
            saliency_map = compute_saliency_map(model, image, threshold=saliency_threshold)
            
            # Visualize and save
            if save_plots:
                save_path = os.path.join(output_dir, f"sample_{idx+1}.png")
            else:
                save_path = None
                
            visualize_saliency(
                image, 
                mask, 
                saliency_map, 
                prediction,
                save_path=save_path,
                show_plot=show_plots,
                prediction_threshold=prediction_threshold
            )
            
            # Save overlay
            if save_plots:
                normalized_saliency = normalize_saliency_map(saliency_map)
                overlay = overlay_saliency_map(
                    image, 
                    normalized_saliency, 
                    alpha=overlay_alpha,
                    colormap=colormap
                )
                
                overlay_path = os.path.join(output_dir, f"sample_{idx+1}_overlay.png")
                plt.figure(figsize=(8, 8))
                plt.imshow(overlay)
                plt.title('Saliency Overlay')
                plt.axis('off')
                plt.savefig(overlay_path)
                if not show_plots:
                    plt.close()
                
                print(f"Processed sample {idx+1}, saved to {save_path} and {overlay_path}")

if __name__ == "__main__":
    # Run with default parameters when script is executed directly
    run_saliency_analysis()