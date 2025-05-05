"""
Waste Classification Dataset Preparation
Author: Zeyad-Diaa-1242
Last updated: 2025-05-05 14:58:30
"""

import os
import glob
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def prepare_waste_dataset(dataset_path, output_dir="./labels", split_ratio=(0.7, 0.15, 0.15)):
    """
    Prepare the waste classification dataset by creating train/validation/test splits.
    
    Args:
        dataset_path: Path to the dataset
        output_dir: Directory to save the split files
        split_ratio: Ratio of train/validation/test split (default: 70/15/15)
    
    Returns:
        Dictionary containing dataset information
    """
    print(f"Preparing waste classification dataset from: {dataset_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store dataset information
    dataset_info = {
        'categories': [],
        'train_data': [],
        'train_labels': [],
        'val_data': [],
        'val_labels': [],
        'test_data': [],
        'test_labels': []
    }
    
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(os.path.join(dataset_path, 'classes')) 
                         if os.path.isdir(os.path.join(dataset_path, 'classes', d))])
    
    print(f"Found {len(class_dirs)} classes: {', '.join(class_dirs)}")
    dataset_info['categories'] = class_dirs
    
    # Collect all image paths and their labels
    all_images = []
    all_labels = []
    
    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(dataset_path, 'classes', class_name)
        
        # Process default images
        default_images = glob.glob(os.path.join(class_path, 'default', '*.png'))
        all_images.extend(default_images)
        all_labels.extend([class_idx] * len(default_images))
        
        # Process real_world images
        real_world_images = glob.glob(os.path.join(class_path, 'real_world', '*.png'))
        all_images.extend(real_world_images)
        all_labels.extend([class_idx] * len(real_world_images))
        
        print(f"  Class {class_name}: {len(default_images)} default images, {len(real_world_images)} real-world images")
    
    # Split data into train/val/test sets
    train_ratio, val_ratio, test_ratio = split_ratio
    
    # First split: separate train and temp (val+test)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        all_images, all_labels, 
        test_size=(val_ratio + test_ratio),
        stratify=all_labels,
        random_state=42
    )
    
    # Second split: separate val and test from temp
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels,
        test_size=test_ratio/(val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=42
    )
    
    # Store split data
    dataset_info['train_data'] = train_images
    dataset_info['train_labels'] = train_labels
    dataset_info['val_data'] = val_images
    dataset_info['val_labels'] = val_labels
    dataset_info['test_data'] = test_images
    dataset_info['test_labels'] = test_labels
    
    print(f"\nDataset split complete:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    # Write split files
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for img_path, label in zip(train_images, train_labels):
            f.write(f"{img_path} {label}\n")
    
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for img_path, label in zip(val_images, val_labels):
            f.write(f"{img_path} {label}\n")
    
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for img_path, label in zip(test_images, test_labels):
            f.write(f"{img_path} {label}\n")
    
    # Write classes file
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for class_name in class_dirs:
            f.write(f"{class_name}\n")
    
    print(f"Dataset files written to {output_dir}")
    
    return dataset_info

# For direct execution
if __name__ == "__main__":
    dataset_path = "/home/x/Desktop/cv2/waste_classification/"
    prepare_waste_dataset(dataset_path)