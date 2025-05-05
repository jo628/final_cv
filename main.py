"""

Waste Classification Pipeline

Author: Zeyad-Diaa-1242

Last updated: 2025-05-05 15:05:10

"""



import os

import cv2

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from tqdm import tqdm

from sklearn.utils import shuffle

import time

import gc
import pickle  # Garbage collector for memory management

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, classification_report



# Import custom modules

from utils import load_dataset, visualize_results, save_model, ModelEvaluator

from preprocessing import ImagePreprocessor

from feature_extraction import FeatureExtractor

from classification import CustomClassifier



# Force CPU usage

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# Set random seeds for reproducibility

np.random.seed(42)

tf.random.set_seed(42)



def main():

    # Confirm we're using CPU

    print("TensorFlow devices available:", tf.config.list_physical_devices())

    print("Is GPU available:", tf.config.list_physical_devices('GPU'))

    print("Using CPU for computation")

    

    print("Starting Waste Classification Pipeline...")

    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")

    

    # 1. Load and prepare dataset

    print("\n[1] Loading dataset...")

    dataset_path = "/home/x/Desktop/cv2/waste_classification/"

    dataset = load_dataset(dataset_path)

    

    if dataset is None:

        print("Error loading dataset. Exiting.")

        return

    

    # Get category names

    categories = dataset['categories']

    print(f"Categories: {categories}")

    print(f"Number of training samples: {len(dataset['train_data'])}")

    print(f"Number of validation samples: {len(dataset['val_data'])}")

    print(f"Number of test samples: {len(dataset['test_data'])}")

    

    # Analyze class distribution

    def analyze_class_distribution(labels, name, categories):

        counts = np.zeros(len(categories), dtype=int)

        for label in labels:

            counts[label] += 1

        

        print(f"\n{name} class distribution:")

        for i, count in enumerate(counts):

            print(f"  Class {categories[i]}: {count} samples ({count/len(labels)*100:.1f}%)")

        return counts

    

    train_counts = analyze_class_distribution(dataset['train_labels'], "Training", categories)

    val_counts = analyze_class_distribution(dataset['val_labels'], "Validation", categories)

    test_counts = analyze_class_distribution(dataset['test_labels'], "Test", categories)

    

    # Use all available data

    train_data = dataset['train_data']

    train_labels = dataset['train_labels']

    val_data = dataset['val_data']

    val_labels = dataset['val_labels']

    test_data = dataset['test_data']

    test_labels = dataset['test_labels']

    

    print(f"\nFinal dataset sizes:")

    print(f"  Training: {len(train_data)} samples")

    print(f"  Validation: {len(val_data)} samples")

    print(f"  Test: {len(test_data)} samples")

    

    # Memory management for large datasets

    print("\nWarning: Processing the full dataset on CPU may take a significant amount of time.")

    print("Starting preprocessing with material-specific augmentations...")

    

    # 2. Preprocessing

    print("\n[2] Applying enhanced preprocessing...")

    

    # Create preprocessor

    preprocessor = ImagePreprocessor(

        target_size=(224, 224), 

        apply_augmentation=True

    )

    

    # Process data in larger batches for the waste dataset

    batch_size = 1500  # Increased batch size as requested

    

    def process_in_batches(data_list, with_augmentation=False, desc="Processing"):

        """Process images in batches to avoid memory overflow"""

        preprocessor.apply_augmentation = with_augmentation

        all_processed = []

        

        for i in range(0, len(data_list), batch_size):

            batch_end = min(i + batch_size, len(data_list))

            batch = data_list[i:batch_end]

            

            print(f"Processing batch {i//batch_size + 1}/{(len(data_list)-1)//batch_size + 1} ({i}:{batch_end})")

            processed_batch = preprocessor.batch_preprocess(batch, segment=False)

            all_processed.extend(processed_batch)

            

            # Force garbage collection to free memory

            gc.collect()

        

        return all_processed

    

    # Process training data with augmentation

    X_train_preprocessed = process_in_batches(train_data, with_augmentation=True, desc="Preprocessing training images")

    

    # Process validation and test data without augmentation

    X_val_preprocessed = process_in_batches(val_data, with_augmentation=False, desc="Preprocessing validation images")

    X_test_preprocessed = process_in_batches(test_data, with_augmentation=False, desc="Preprocessing test images")

    

    # 3. Enhanced Feature Extraction

    print("\n[3] Extracting features...")

    

    # Initialize feature extractor

    feature_extractor = FeatureExtractor(input_shape=(224, 224, 3))

    

    # Extract features in small batches for stability

    def extract_features_in_batches(images, batch_size=4, desc="Extracting features"):

        """Extract features in smaller batches for better stability"""

        all_features = []

        

        for i in range(0, len(images), batch_size):

            batch_end = min(i + batch_size, len(images))

            batch = images[i:batch_end]

            

            print(f"Extracting features batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1} ({i}:{batch_end})")

            try:

                batch_features = feature_extractor.batch_extract_features(batch, batch_size=4)

                all_features.extend(batch_features)

            except Exception as e:

                print(f"Error in batch extraction: {e}")

                # Create placeholder features

                placeholder_features = [np.zeros(1500) for _ in range(len(batch))]

                all_features.extend(placeholder_features)

            

            # Force garbage collection

            gc.collect()

        

        return np.array(all_features)

    

    # Extract features with careful error handling

    X_train_features = extract_features_in_batches(X_train_preprocessed, batch_size=4)

    X_val_features = extract_features_in_batches(X_val_preprocessed, batch_size=4)

    X_test_features = extract_features_in_batches(X_test_preprocessed, batch_size=4)

    

    # Convert labels to numpy arrays

    train_labels = np.array(train_labels)

    val_labels = np.array(val_labels)

    test_labels = np.array(test_labels)

    

    print(f"Training features shape: {X_train_features.shape}")

    print(f"Validation features shape: {X_val_features.shape}")

    print(f"Test features shape: {X_test_features.shape}")

    

    # 4. Advanced Feature Preprocessing

    print("\n[4] Applying feature preprocessing...")

    

    # Replace any NaN or infinite values

    X_train_features = np.nan_to_num(X_train_features)

    X_val_features = np.nan_to_num(X_val_features)

    X_test_features = np.nan_to_num(X_test_features)

    

    # Feature normalization

    scaler = StandardScaler()

    X_train_processed = scaler.fit_transform(X_train_features)

    X_val_processed = scaler.transform(X_val_features)

    X_test_processed = scaler.transform(X_test_features)

    

    print(f"Processed feature shapes:")

    print(f"  Training: {X_train_processed.shape}")

    print(f"  Validation: {X_val_processed.shape}")

    print(f"  Test: {X_test_processed.shape}")

    

    # 5. Enhanced Model Training with Anti-Overfitting Measures

    print("\n[5] Training model with anti-overfitting measures...")

    

    # Calculate class weights

    class_counts = np.bincount(train_labels)

    total_samples = len(train_labels)

    

    # Create balanced weights for all classes

    class_weights = {}

    for i in range(len(categories)):

        if i in class_counts and class_counts[i] > 0:

            class_weights[i] = total_samples / (len(categories) * class_counts[i])

        else:

            class_weights[i] = 1.0

    

    print(f"Class weights: {class_weights}")

    

    # Create classifier with strong anti-overfitting measures

    classifier = CustomClassifier(

        input_shape=X_train_processed.shape[1],  # Feature dimension

        num_classes=len(categories),             # Number of classes (now 6 instead of 3)

        learning_rate=0.0005,                    # Lower learning rate

        dropout_rate=0.6,                        # Higher dropout

        l2_reg=0.02                              # Stronger regularization

    )

    

    # Use a max_samples parameter to limit training samples if needed

    max_samples = min(4000, len(X_train_processed))  # Limit samples for very large datasets

    

    # Train with anti-overfitting strategy

    history = classifier.train(

        X_train=X_train_processed,

        y_train=train_labels,

        X_val=X_val_processed,

        y_val=val_labels,

        batch_size=64,                           # Larger batch size for larger dataset

        epochs=30,                               # Moderate number of epochs

        patience=8,                              # Early stopping

        class_weights=class_weights,             # Balanced class weights

        max_samples=max_samples                  # Limit number of samples used in training

    )

    

    # Plot training history

    classifier.plot_training_history(history)

    

    # 6. Robust Evaluation

    print("\n[6] Evaluating model...")

    evaluator = ModelEvaluator(categories)

    

    # Make predictions

    y_pred = classifier.predict(X_test_processed)

    

    # Get probabilities directly 

    probabilities = classifier.model.predict(X_test_processed, verbose=0)

    

    # Check distribution of predictions

    unique_preds, pred_counts = np.unique(y_pred, return_counts=True)

    print(f"Prediction distribution:")

    for pred_class in unique_preds:

        count = pred_counts[pred_class]

        print(f"  Class {categories[pred_class]}: {count} predictions ({count/len(y_pred)*100:.1f}%)")

    

    # Calculate and print metrics

    evaluator.print_metrics(test_labels, y_pred, dataset_name="Test")

    

    # Print confusion matrix as text

    print("\nConfusion Matrix:")

    cm = confusion_matrix(test_labels, y_pred)

    print(cm)

    

    # Plot confusion matrix

    try:

        evaluator.plot_confusion_matrix(test_labels, y_pred)

    except Exception as e:

        print(f"Could not plot confusion matrix: {e}")

    

    # 7. Class-Specific Analysis

    print("\n[7] Class-specific analysis...")

    

    # Per-class metrics

    class_report = classification_report(test_labels, y_pred, target_names=categories, output_dict=True)

    

    # Print detailed per-class metrics

    print("\nDetailed metrics by class:")

    for material in categories:

        metrics = class_report[material]

        print(f"  {material.capitalize()}:")

        print(f"    Precision: {metrics['precision']:.4f}")

        print(f"    Recall: {metrics['recall']:.4f}")

        print(f"    F1-score: {metrics['f1-score']:.4f}")

        print(f"    Support: {metrics['support']} samples")

    

    # 8. Display Example Predictions

    print("\n[8] Analyzing interesting examples...")

    

    # Get some correctly and incorrectly classified samples

    correct_indices = np.where(y_pred == test_labels)[0]

    incorrect_indices = np.where(y_pred != test_labels)[0]

    

    # Sample selection

    num_examples = min(5, len(incorrect_indices))

    

    if len(incorrect_indices) > 0:

        # Select most confidently misclassified examples

        confidences = np.max(probabilities[incorrect_indices], axis=1)

        most_confident_incorrect = incorrect_indices[np.argsort(confidences)[-num_examples:]]

        

        print("Most confidently misclassified examples:")

        test_samples = []

        for idx in most_confident_incorrect:

            img = cv2.imread(test_data[idx])

            img = cv2.resize(img, (224, 224))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            test_samples.append(img)

            

            # Print details

            true_label = categories[test_labels[idx]]

            pred_label = categories[y_pred[idx]]

            confidence = np.max(probabilities[idx]) * 100

            print(f"  Image {idx}: True={true_label}, Predicted={pred_label}, Confidence={confidence:.1f}%")

        

        # Show examples if visualization function exists

        try:

            evaluator.plot_examples(

                test_samples,

                test_labels[most_confident_incorrect],

                y_pred[most_confident_incorrect],

                num_examples=len(test_samples),

                title="Most Confidently Misclassified Waste Items"

            )

        except Exception as e:

            print(f"Could not visualize examples: {e}")

    

    # 9. Save finalized model

    print("\n[9] Saving model...")

    save_model(classifier.model, "waste_classifier_anti_overfitting.h5")

    print("Model saved to waste_classifier_anti_overfitting.h5")

    

    

    # Save the scaler for use in the GUI

    with open('feature_scaler.pkl', 'wb') as f:

        pickle.dump(scaler, f)

    print("Feature scaler saved for GUI")


    

    # Save the scaler for use in the GUI

    with open('feature_scaler.pkl', 'wb') as f:

        pickle.dump(scaler, f)

    print("Feature scaler saved for GUI")


    

    # Save the scaler for use in the GUI

    with open('feature_scaler.pkl', 'wb') as f:

        pickle.dump(scaler, f)

    print("Feature scaler saved for GUI")


    

    # Save the scaler for use in the GUI

    with open('feature_scaler.pkl', 'wb') as f:

        pickle.dump(scaler, f)

    print("Feature scaler saved for GUI")


    print("\nWaste Classification Pipeline completed successfully!")



if __name__ == "__main__":

    main()