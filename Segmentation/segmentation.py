import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import random
import pickle

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)



# Define a custom transform class for paired image and mask
class PairedTransform:
    def __init__(self, blur_prob=0.5, rotation_prob=0.5, blur_kernel=(5, 5)):
        self.blur_prob = blur_prob
        self.rotation_prob = rotation_prob
        self.blur_kernel = blur_kernel
        
    def __call__(self, image, mask):
        # Convert tensors to numpy for OpenCV operations
        # Apply blur randomly only to the image (not the mask)
        if random.random() < self.blur_prob:
            image_np = image.squeeze(0).numpy()
            image_np = cv2.GaussianBlur(image_np, self.blur_kernel, 0)
            image = torch.from_numpy(image_np).unsqueeze(0)
        
        # Apply same rotation to both image and mask
        if random.random() < self.rotation_prob:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
            
        return image, mask



# Data Preprocessing
def load_and_preprocess_data(base_dir):
    # Get a list of all patient folders
    patient_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) 
                      and f not in ['__pycache__']]
    
    images = []
    masks = []
    
    for patient_folder in tqdm(patient_folders):
        patient_path = os.path.join(base_dir, patient_folder)
        
        # Find all image files and their corresponding masks
        image_files = [f for f in glob(os.path.join(patient_path, "**/*"), recursive=True) 
                      if f.endswith(('.png', '.jpg', '.jpeg', '.tif')) and 'mask' not in f.lower()]
        
        for img_path in image_files:
            # Assuming mask file is in the same folder with a pattern like "[image_name]_mask.[ext]"
            base_name = os.path.splitext(img_path)[0]
            mask_path = f"{base_name}_mask{os.path.splitext(img_path)[1]}"
            
            if os.path.exists(mask_path):
                # Load and preprocess image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (256, 256))
                    img = img / 255.0  # Normalize to [0, 1]
                    
                    # Load and preprocess mask
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = cv2.resize(mask, (256, 256))
                        mask = (mask > 0).astype(np.float32)  # Binary mask
                        
                        images.append(img)
                        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Define Dataset class
class BrainMRIDataset(Dataset):
    def __init__(self, images, masks, transform=None, augment=False):
        self.images = images
        self.masks = masks
        self.transform = transform  # Individual transforms for the image only
        self.augment = augment
        self.paired_transform = PairedTransform() if augment else None
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        # Convert to PyTorch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        # Apply paired transformations (blur, rotation) if augmentation is enabled
        if self.paired_transform:
            image, mask = self.paired_transform(image, mask)
            
        # Apply additional transforms only to the image
        if self.transform:
            image = self.transform(image)
            
        return image, mask

# Split data into train, validation, and test sets
def train_val_test_split(images, masks, val_ratio=0.15, test_ratio=0.15):
    n = len(images)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - val_size - test_size
    
    indices = np.random.permutation(n)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_images, train_masks = images[train_indices], masks[train_indices]
    val_images, val_masks = images[val_indices], masks[val_indices] 
    test_images, test_masks = images[test_indices], masks[test_indices]
    
    return (train_images, train_masks), (val_images, val_masks), (test_images, test_masks)

# Define loss functions
def dice_loss(pred, target):
    smooth = 1.0
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice_score

def combined_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    
    return bce_weight * bce + (1 - bce_weight) * dice

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler=None, num_epochs=100, patience=10, device='cuda', model_name="unet"):
    model.to(device)
    best_val_loss = float('inf')
    patient_counter = 0
    best_model_path = f'best_{model_name}_model.pt'
    
    # Initialize history dict to store metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate dice for tracking
            with torch.no_grad():
                dice = dice_loss(outputs, masks)
                train_dice += dice.item()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average metrics for the epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += (dice_loss(outputs, masks)).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        # Update learning rate if scheduler exists
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(avg_val_dice)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        
        # Save best model and check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            # Save the full model
            torch.save(model, f'best_{model_name}_full.pt')
            # Save using TorchScript for deployment
            scripted_model = torch.jit.script(model)
            scripted_model.save(f'best_{model_name}_scripted.pt')
            # Save as pickle file
            with open(f'best_{model_name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            patient_counter = 0
        else:
            patient_counter += 1
            print(f"EarlyStopping counter: {patient_counter} out of {patience}")
            if patient_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    return model, history

# Visualization function
def visualize_results(model, test_loader, num_samples=5, device='cuda', model_name="unet"):
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Move to CPU for visualization
            image = images[0, 0].cpu().numpy()
            mask = masks[0, 0].cpu().numpy()
            pred = outputs[0, 0].cpu().numpy() > 0.5  # Apply threshold
            
            # Plot results
            axes[i, 0].imshow(image, cmap='gray')
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_results.png')
    plt.show()

# Function to calculate metrics on test set
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    test_dice = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred = (outputs > 0.5).float()  # Apply threshold
            
            # Calculate dice score
            dice = 1 - dice_loss(pred, masks)
            test_dice += dice.item()
    
    avg_test_dice = test_dice / len(test_loader)
    print(f"Test Dice Score: {avg_test_dice:.4f}")
    
    return avg_test_dice

# Main execution pipeline
def run_segmentation_pipeline(model, base_dir="kaggle_3m", model_name="unet", batch_size=8, patience=10):
    """
    Run the complete segmentation pipeline with a custom model.
    
    Args:
        model: A PyTorch model instance (should take single-channel input and produce single-channel output)
        base_dir: Directory containing the dataset
        model_name: String identifier for the model (used for file naming)
        batch_size: Batch size for training
        
    Returns:
        model: Trained model
        test_dice: Evaluation metric on test set
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading data...")
    images, masks = load_and_preprocess_data(base_dir)
    
    # Split data
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = train_val_test_split(images, masks)
    
    print(f"Train: {len(train_images)}, Validation: {len(val_images)}, Test: {len(test_images)}")
    
    # Create datasets
# Create datasets with augmentation for training
    train_dataset = BrainMRIDataset(train_images, train_masks, augment=True)
    val_dataset = BrainMRIDataset(val_images, val_masks, augment=False)
    test_dataset = BrainMRIDataset(test_images, test_masks, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5)
    
    # Define loss function (50% BCE + 50% Dice)
    criterion = lambda pred, target: combined_loss(pred, target, bce_weight=0.5)
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        patience=patience,
        device=device,
        model_name=model_name
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice'], label='Train')
    plt.plot(history['val_dice'], label='Validation')
    plt.title('Dice Score')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.show()
    
    # Evaluate on test set
    test_dice = evaluate_model(model, test_loader, device)
    
    # Visualize results
    visualize_results(model, test_loader, num_samples=5, device=device, model_name=model_name)
    
    return model, test_dice

if __name__ == "__main__":
    # Example usage - you need to define your own model
    pass




'''# Import the utilities
from segmentation import run_segmentation_pipeline

# Define your custom U-Net or any segmentation model here
class MyCustomUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(MyCustomUNet, self).__init__()
        # Define your custom architecture here
        # ...

    def forward(self, x):
        # Define the forward pass
        # ...
        return output

# Create your model instance
my_model = MyCustomUNet(in_channels=1, out_channels=1)

# Run the pipeline with custom model and naming
trained_model, dice_score = run_segmentation_pipeline(
    model=my_model,
    base_dir="path/to/dataset",
    model_name="my_custom_unet",
    batch_size=8
)'''