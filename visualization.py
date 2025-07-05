"""
Visualization module for CIFAR-10 training and inference
Provides comprehensive plotting functions for training history, model performance, and data analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import os
import json
from config import Config, CIFAR10_CLASSES, CLASS_COLORS

class TrainingVisualizer:
    """Visualization class for training history and model performance"""
    
    def __init__(self, config: Config):
        """
        Initialize visualizer
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.class_names = CIFAR10_CLASSES
        self.class_colors = CLASS_COLORS
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_history(
        self, 
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot training history
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        # Plot loss
        axes[0].plot(history.get('loss', []), label='Training Loss', color='blue')
        axes[0].plot(history.get('val_loss', []), label='Validation Loss', color='red')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(history.get('accuracy', []), label='Training Accuracy', color='blue')
        axes[1].plot(history.get('val_accuracy', []), label='Validation Accuracy', color='red')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot top-3 accuracy if available
        if 'top_3_accuracy' in history:
            axes[2].plot(history.get('top_3_accuracy', []), label='Training Top-3 Accuracy', color='blue')
            axes[2].plot(history.get('val_top_3_accuracy', []), label='Validation Top-3 Accuracy', color='red')
            axes[2].set_title('Top-3 Accuracy')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Top-3 Accuracy')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Top-3 Accuracy\nNot Available', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Top-3 Accuracy')
        
        # Plot learning rate if available
        if 'learning_rate' in history:
            axes[3].plot(history.get('learning_rate', []), color='green')
            axes[3].set_title('Learning Rate')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Learning Rate')
            axes[3].set_yscale('log')
            axes[3].grid(True, alpha=0.3)
        else:
            axes[3].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Learning Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(
        self, 
        confusion_matrix: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            save_path: Path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_class_accuracy(
        self, 
        class_accuracies: Dict[str, float],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot per-class accuracy
        
        Args:
            class_accuracies: Dictionary of class accuracies
            save_path: Path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Prepare data
        classes = list(class_accuracies.keys())
        accuracies = list(class_accuracies.values())
        colors = [self.class_colors.get(cls, '#666666') for cls in classes]
        
        # Create bar plot
        bars = plt.bar(classes, accuracies, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class accuracy plot saved to {save_path}")
        
        plt.show()
    
    def plot_confidence_distribution(
        self, 
        confidences: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot confidence distribution
        
        Args:
            confidences: Prediction confidences
            save_path: Path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create histogram
        plt.hist(confidences, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        
        # Add mean line
        mean_conf = np.mean(confidences)
        plt.axvline(mean_conf, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_conf:.3f}')
        
        # Add median line
        median_conf = np.median(confidences)
        plt.axvline(median_conf, color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {median_conf:.3f}')
        
        plt.title('Confidence Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence distribution saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_samples(
        self, 
        images: np.ndarray,
        predicted_classes: np.ndarray,
        confidences: np.ndarray,
        true_classes: Optional[np.ndarray] = None,
        num_samples: int = 16,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 15)
    ):
        """
        Plot sample predictions
        
        Args:
            images: Input images
            predicted_classes: Predicted class labels
            confidences: Prediction confidences
            true_classes: True class labels (optional)
            num_samples: Number of samples to display
            save_path: Path to save the plot
            figsize: Figure size
        """
        # Limit number of samples
        num_samples = min(num_samples, len(images))
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        
        for i in range(grid_size * grid_size):
            if i < num_samples:
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
                    axes[i].set_title(title, color=color, fontsize=10, fontweight='bold')
                else:
                    axes[i].set_title(title, fontsize=10, fontweight='bold')
                
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction samples saved to {save_path}")
        
        plt.show()
    
    def plot_training_metrics_comparison(
        self, 
        histories: Dict[str, Dict[str, List[float]]],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot comparison of training metrics from different models/runs
        
        Args:
            histories: Dictionary of training histories
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        # Plot training loss comparison
        for name, history in histories.items():
            axes[0].plot(history.get('loss', []), label=f'{name} - Train', alpha=0.7)
            axes[0].plot(history.get('val_loss', []), label=f'{name} - Val', linestyle='--', alpha=0.7)
        
        axes[0].set_title('Training Loss Comparison')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot training accuracy comparison
        for name, history in histories.items():
            axes[1].plot(history.get('accuracy', []), label=f'{name} - Train', alpha=0.7)
            axes[1].plot(history.get('val_accuracy', []), label=f'{name} - Val', linestyle='--', alpha=0.7)
        
        axes[1].set_title('Training Accuracy Comparison')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot final metrics comparison
        final_metrics = {}
        for name, history in histories.items():
            final_metrics[name] = {
                'final_train_acc': history.get('accuracy', [0])[-1] if history.get('accuracy') else 0,
                'final_val_acc': history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 0,
                'final_train_loss': history.get('loss', [0])[-1] if history.get('loss') else 0,
                'final_val_loss': history.get('val_loss', [0])[-1] if history.get('val_loss') else 0
            }
        
        # Bar plot for final accuracies
        names = list(final_metrics.keys())
        train_accs = [final_metrics[name]['final_train_acc'] for name in names]
        val_accs = [final_metrics[name]['final_val_acc'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        axes[2].bar(x - width/2, train_accs, width, label='Training', alpha=0.7)
        axes[2].bar(x + width/2, val_accs, width, label='Validation', alpha=0.7)
        axes[2].set_title('Final Accuracy Comparison')
        axes[2].set_xlabel('Model')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, rotation=45)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Bar plot for final losses
        train_losses = [final_metrics[name]['final_train_loss'] for name in names]
        val_losses = [final_metrics[name]['final_val_loss'] for name in names]
        
        axes[3].bar(x - width/2, train_losses, width, label='Training', alpha=0.7)
        axes[3].bar(x + width/2, val_losses, width, label='Validation', alpha=0.7)
        axes[3].set_title('Final Loss Comparison')
        axes[3].set_xlabel('Model')
        axes[3].set_ylabel('Loss')
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(names, rotation=45)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training metrics comparison saved to {save_path}")
        
        plt.show()
    
    def create_training_report(
        self, 
        history: Dict[str, List[float]],
        evaluation_results: Dict[str, Any],
        save_dir: str = "results"
    ):
        """
        Create a comprehensive training report with all visualizations
        
        Args:
            history: Training history
            evaluation_results: Evaluation results
            save_dir: Directory to save the report
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create training history plot
        self.plot_training_history(
            history, 
            save_path=os.path.join(save_dir, 'training_history.png')
        )
        
        # Create confusion matrix if available
        if 'confusion_matrix' in evaluation_results:
            cm = np.array(evaluation_results['confusion_matrix'])
            self.plot_confusion_matrix(
                cm, 
                save_path=os.path.join(save_dir, 'confusion_matrix.png')
            )
        
        # Create class accuracy plot if available
        if 'per_class_accuracy' in evaluation_results:
            self.plot_class_accuracy(
                evaluation_results['per_class_accuracy'],
                save_path=os.path.join(save_dir, 'class_accuracy.png')
            )
        
        # Create confidence distribution if available
        if 'confidence_stats' in evaluation_results:
            # This would need actual confidence values, not just stats
            # For now, we'll skip this
            pass
        
        print(f"Training report saved to {save_dir}/")

def create_visualizer(config: Config) -> TrainingVisualizer:
    """
    Create and return a training visualizer
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized visualizer
    """
    return TrainingVisualizer(config) 