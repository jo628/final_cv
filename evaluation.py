import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class ModelEvaluator:
    def __init__(self, categories):
        """
        Initialize the model evaluator.
        
        Args:
            categories: List of category names
        """
        self.categories = categories
        self.num_classes = len(categories)
        
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        # Check if we have multiple classes in predictions
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        # Only compute these metrics if we have multiple classes in both true and predicted
        if len(unique_true) > 1 and len(unique_pred) > 1:
            metrics.update({
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            })
        else:
            metrics.update({
                'precision': 0.0 if len(unique_pred) <= 1 else 1.0,
                'recall': 0.0 if len(unique_pred) <= 1 else 1.0,
                'f1': 0.0 if len(unique_pred) <= 1 else 1.0
            })
            
        return metrics
    
    def print_metrics(self, y_true, y_pred, dataset_name="Test"):
        """
        Print classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            dataset_name: Name of the dataset (e.g., "Test", "Validation")
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        
        print(f"\n===== {dataset_name} Set Metrics =====")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Get unique classes in the data
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        present_classes = sorted(list(set(unique_true) | set(unique_pred)))
        
        # Print per-class metrics if possible
        if len(unique_pred) > 1:
            try:
                print("\nClassification Report:")
                # Only use labels that are present in the data
                target_names = [self.categories[i] for i in present_classes]
                report = classification_report(
                    y_true, y_pred, 
                    labels=present_classes, 
                    target_names=target_names,
                    zero_division=0
                )
                print(report)
            except Exception as e:
                print(f"Could not generate classification report: {e}")
                print(f"Unique true labels: {unique_true}")
                print(f"Unique predicted labels: {unique_pred}")
        else:
            print("\nCannot generate classification report: all predictions are the same class")
            print(f"Predicted class: {self.categories[unique_pred[0]] if len(unique_pred) > 0 else 'None'}")
            class_counts = np.bincount(y_true, minlength=self.num_classes)
            print("True label distribution:")
            for i, count in enumerate(class_counts):
                if count > 0:
                    print(f"  {self.categories[i]}: {count} samples")
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True, figsize=(10, 8)):
        """
        Plot the confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
        """
        # Check if we have multiple classes in predictions
        unique_pred = np.unique(y_pred)
        if len(unique_pred) <= 1:
            print("Cannot generate confusion matrix: all predictions are the same class")
            return
            
        # Get unique classes in the data
        unique_true = np.unique(y_true)
        present_classes = sorted(list(set(unique_true) | set(unique_pred)))
        
        # Compute confusion matrix for only the classes present in the data
        cm = confusion_matrix(y_true, y_pred, labels=present_classes)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'
        
        # Plot confusion matrix
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f' if normalize else 'd', 
            cmap='Blues',
            xticklabels=[self.categories[i] for i in present_classes],
            yticklabels=[self.categories[i] for i in present_classes]
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_examples(self, images, y_true, y_pred, num_examples=5):
        """
        Plot examples of correct and incorrect predictions.
        
        Args:
            images: List of images
            y_true: True labels
            y_pred: Predicted labels
            num_examples: Number of examples to show
        """
        # Find correct and incorrect predictions
        correct = np.where(y_true == y_pred)[0]
        incorrect = np.where(y_true != y_pred)[0]
        
        # Plot correct predictions
        if len(correct) > 0:
            n = min(num_examples, len(correct))
            indices = np.random.choice(correct, n, replace=False)
            
            plt.figure(figsize=(15, 3))
            plt.suptitle("Correct Predictions", fontsize=14)
            
            for i, idx in enumerate(indices):
                plt.subplot(1, n, i + 1)
                
                # Convert image for display
                img = images[idx]
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                
                if len(img.shape) == 3 and img.shape[2] == 3:
                    plt.imshow(img)
                else:
                    plt.imshow(img, cmap='gray')
                    
                plt.title(f"True: {self.categories[y_true[idx]]}")
                plt.axis('off')
                
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        
        # Plot incorrect predictions
        if len(incorrect) > 0:
            n = min(num_examples, len(incorrect))
            indices = np.random.choice(incorrect, n, replace=False)
            
            plt.figure(figsize=(15, 3))
            plt.suptitle("Incorrect Predictions", fontsize=14)
            
            for i, idx in enumerate(indices):
                plt.subplot(1, n, i + 1)
                
                # Convert image for display
                img = images[idx]
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                
                if len(img.shape) == 3 and img.shape[2] == 3:
                    plt.imshow(img)
                else:
                    plt.imshow(img, cmap='gray')
                    
                plt.title(f"True: {self.categories[y_true[idx]]}\nPred: {self.categories[y_pred[idx]]}")
                plt.axis('off')
                
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()