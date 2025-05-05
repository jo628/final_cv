"""
Utility Functions for Waste Classification
Author: Zeyad-Diaa-1242
Last updated: 2025-05-05 15:02:45
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf

def load_dataset(dataset_path, labels_dir="./labels", fold=None):
    """
    Load the waste classification dataset.
    
    Args:
        dataset_path: Path to the dataset
        labels_dir: Path to labels directory (default: ./labels)
        fold: Ignored parameter for compatibility
    
    Returns:
        Dictionary containing dataset information
    """
    # Check if the labels directory exists, if not create it and prepare the dataset
    if not os.path.exists(labels_dir) or not os.listdir(labels_dir):
        try:
            from prepare_dataset import prepare_waste_dataset
            print("Labels not found. Preparing dataset...")
            return prepare_waste_dataset(dataset_path, labels_dir)
        except ImportError:
            print("Error: prepare_dataset module not found. Please run prepare_dataset.py first.")
            return None
    
    # Dictionary to store dataset information
    dataset = {
        'categories': [],
        'train_data': [],
        'train_labels': [],
        'val_data': [],
        'val_labels': [],
        'test_data': [],
        'test_labels': []
    }
    
    # Load class names
    with open(os.path.join(labels_dir, 'classes.txt'), 'r') as f:
        dataset['categories'] = [line.strip() for line in f.readlines()]
    
    # Load training data
    with open(os.path.join(labels_dir, 'train.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_path = parts[0]
                label = int(parts[1])
                
                dataset['train_data'].append(img_path)
                dataset['train_labels'].append(label)
    
    # Load validation data
    with open(os.path.join(labels_dir, 'val.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_path = parts[0]
                label = int(parts[1])
                
                dataset['val_data'].append(img_path)
                dataset['val_labels'].append(label)
    
    # Load test data
    with open(os.path.join(labels_dir, 'test.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_path = parts[0]
                label = int(parts[1])
                
                dataset['test_data'].append(img_path)
                dataset['test_labels'].append(label)
    
    return dataset

def visualize_results(images, true_labels, pred_labels, class_names, num_examples=5):
    """
    Visualize prediction results.
    
    Args:
        images: List of images
        true_labels: List of true labels
        pred_labels: List of predicted labels
        class_names: List of class names
        num_examples: Number of examples to visualize
    """
    plt.figure(figsize=(15, 10))
    
    for i in range(min(num_examples, len(images))):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

def save_model(model, model_path):
    """
    Save trained model.
    
    Args:
        model: Trained model
        model_path: Path to save the model
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")

class ModelEvaluator:
    """Class for evaluating classification models"""
    
    def __init__(self, class_names):
        """Initialize with class names"""
        self.class_names = class_names
    
    def print_metrics(self, y_true, y_pred, dataset_name="Test"):
        """Print classification metrics"""
        accuracy = np.mean(y_true == y_pred)
        
        # Calculate precision, recall for each class
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # Calculate overall precision, recall, f1 (macro)
        precision = np.mean([report[class_name]['precision'] for class_name in self.class_names])
        recall = np.mean([report[class_name]['recall'] for class_name in self.class_names])
        f1 = np.mean([report[class_name]['f1-score'] for class_name in self.class_names])
        
        print(f"\n===== {dataset_name} Set Metrics =====")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                    cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_examples(self, images, true_labels, pred_labels, num_examples=5, title=None):
        """Visualize examples with predictions"""
        plt.figure(figsize=(15, 10))
        
        for i in range(min(num_examples, len(images))):
            plt.subplot(1, num_examples, i + 1)
            plt.imshow(images[i])
            
            true_class = self.class_names[true_labels[i]]
            pred_class = self.class_names[pred_labels[i]]
            
            color = 'green' if true_class == pred_class else 'red'
            
            plt.title(f"True: {true_class}\nPred: {pred_class}", color=color)
            plt.axis('off')
        
        if title:
            plt.suptitle(title, fontsize=16)
            
        plt.tight_layout()
        plt.show()