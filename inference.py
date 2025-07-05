"""
Inference module for CIFAR-10 classification
Handles model loading, prediction, and result visualization
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
from config import Config, CIFAR10_CLASSES, CLASS_COLORS

class CIFAR10Inference:
    """Inference class for CIFAR-10 classification"""
    
    def __init__(self, config: Config, model_path: Optional[str] = None):
        """
        Initialize inference engine
        
        Args:
            config: Configuration object
            model_path: Path to the trained model (optional)
        """
        self.config = config
        self.model = None
        self.class_names = CIFAR10_CLASSES
        self.class_colors = CLASS_COLORS
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a trained model
        
        Args:
            model_path: Path to the saved model
        """
        print(f"Loading model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
            
            # Print model summary
            print("\nModel Summary:")
            self.model.summary()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image for inference
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        # Ensure correct shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Convert to float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Resize if needed
        if image.shape[1:3] != (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE):
            image = tf.image.resize(image, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            image = image.numpy()
        
        return image
    
    def predict_single(self, image: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Make prediction on a single image
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get results
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        all_probabilities = predictions[0]
        
        return predicted_class, confidence, all_probabilities
    
    def predict_batch(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of images
        
        Args:
            images: Batch of input images
            
        Returns:
            Tuple of (predicted_classes, confidences, all_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess images
        processed_images = []
        for image in images:
            processed_image = self.preprocess_image(image)
            processed_images.append(processed_image[0])  # Remove batch dimension
        
        processed_images = np.array(processed_images)
        
        # Make predictions
        predictions = self.model.predict(processed_images, verbose=0)
        
        # Get results
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return predicted_classes, confidences, predictions
    
    def predict_with_confidence_threshold(
        self, 
        images: np.ndarray, 
        threshold: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence threshold filtering
        
        Args:
            images: Batch of input images
            threshold: Confidence threshold (uses config default if None)
            
        Returns:
            Tuple of (predicted_classes, confidences, filtered_indices)
        """
        if threshold is None:
            threshold = self.config.CONFIDENCE_THRESHOLD
        
        # Get predictions
        predicted_classes, confidences, _ = self.predict_batch(images)
        
        # Filter by confidence threshold
        confident_indices = confidences >= threshold
        
        return predicted_classes, confidences, confident_indices
    
    def evaluate_predictions(
        self, 
        true_labels: np.ndarray, 
        predicted_labels: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate prediction results
        
        Args:
            true_labels: True class labels
            predicted_labels: Predicted class labels
            confidences: Prediction confidences (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate accuracy
        accuracy = np.mean(true_labels == predicted_labels)
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = true_labels == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predicted_labels[class_mask] == true_labels[class_mask])
                per_class_accuracy[class_name] = class_accuracy
        
        # Generate classification report
        report = classification_report(
            true_labels, 
            predicted_labels, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Prepare results
        results = {
            "overall_accuracy": accuracy,
            "per_class_accuracy": per_class_accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "num_samples": len(true_labels)
        }
        
        # Add confidence statistics if available
        if confidences is not None:
            results["confidence_stats"] = {
                "mean_confidence": np.mean(confidences),
                "std_confidence": np.std(confidences),
                "min_confidence": np.min(confidences),
                "max_confidence": np.max(confidences)
            }
        
        return results
    
    def visualize_predictions(
        self, 
        images: np.ndarray, 
        predicted_classes: np.ndarray,
        confidences: np.ndarray,
        true_classes: Optional[np.ndarray] = None,
        num_samples: int = 16,
        save_path: Optional[str] = None
    ):
        """
        Visualize prediction results
        
        Args:
            images: Input images
            predicted_classes: Predicted class labels
            confidences: Prediction confidences
            true_classes: True class labels (optional)
            num_samples: Number of samples to visualize
            save_path: Path to save the visualization
        """
        # Limit number of samples
        num_samples = min(num_samples, len(images))
        
        # Create subplot
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.ravel()
        
        for i in range(num_samples):
            # Get image and predictions
            img = images[i]
            pred_class = predicted_classes[i]
            confidence = confidences[i]
            true_class = true_classes[i] if true_classes is not None else None
            
            # Denormalize image if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            # Display image
            axes[i].imshow(img)
            
            # Set title with prediction info
            title = f"Pred: {self.class_names[pred_class]}\nConf: {confidence:.3f}"
            if true_class is not None:
                correct = pred_class == true_class
                color = 'green' if correct else 'red'
                title += f"\nTrue: {self.class_names[true_class]}"
                title += f" ({'✓' if correct else '✗'})"
                axes[i].set_title(title, color=color, fontsize=10)
            else:
                axes[i].set_title(title, fontsize=10)
            
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(
        self, 
        confusion_matrix: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_confidence_distribution(
        self, 
        confidences: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot confidence distribution
        
        Args:
            confidences: Prediction confidences
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence distribution saved to {save_path}")
        
        plt.show()
    
    def save_inference_results(
        self, 
        results: Dict[str, Any], 
        save_path: str = "results/inference_results.json"
    ):
        """
        Save inference results to file
        
        Args:
            results: Inference results dictionary
            save_path: Path to save results
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Inference results saved to {save_path}")
    
    def print_inference_summary(self, results: Dict[str, Any]):
        """Print a summary of inference results"""
        print("\n" + "="*50)
        print("INFERENCE RESULTS SUMMARY")
        print("="*50)
        
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Number of Samples: {results['num_samples']}")
        
        if 'confidence_stats' in results:
            stats = results['confidence_stats']
            print(f"\nConfidence Statistics:")
            print(f"  Mean: {stats['mean_confidence']:.4f}")
            print(f"  Std: {stats['std_confidence']:.4f}")
            print(f"  Min: {stats['min_confidence']:.4f}")
            print(f"  Max: {stats['max_confidence']:.4f}")
        
        print(f"\nPer-Class Accuracy:")
        for class_name, accuracy in results['per_class_accuracy'].items():
            print(f"  {class_name}: {accuracy:.4f}")
        
        print("="*50 + "\n")

def create_inference_engine(config: Config, model_path: Optional[str] = None) -> CIFAR10Inference:
    """
    Create and return an inference engine
    
    Args:
        config: Configuration object
        model_path: Path to the trained model (optional)
        
    Returns:
        Initialized inference engine
    """
    return CIFAR10Inference(config, model_path) 