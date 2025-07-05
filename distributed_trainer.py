"""
Distributed trainer module for CIFAR-10 classification
Handles distributed training across multiple GPUs with proper strategy management
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import time
import datetime
from typing import Dict, Any, Optional, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from data_loader import CIFAR10DataLoader
from models import create_model, print_model_info
from callbacks import create_callbacks, print_callback_info, save_callback_history

class DistributedTrainer:
    """Distributed trainer class for CIFAR-10 classification"""
    
    def __init__(self, config: Config):
        """
        Initialize distributed trainer
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.strategy = None
        self.model = None
        self.data_loader = None
        self.callbacks = None
        self.history = None
        
        # Training metrics
        self.training_time = 0
        self.best_accuracy = 0.0
        self.best_val_accuracy = 0.0
        
        # Initialize components
        self._setup_distributed_strategy()
        self._setup_data_loader()
        self._setup_model()
        self._setup_callbacks()
    
    def _setup_distributed_strategy(self):
        """Setup distributed training strategy"""
        print("Setting up distributed training strategy...")
        
        # Check available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Found {len(gpus)} GPU(s)")
        
        if len(gpus) == 0:
            print("No GPUs found, using CPU")
            self.strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        elif len(gpus) == 1:
            print("Single GPU detected, using OneDeviceStrategy")
            self.strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        else:
            print(f"Multiple GPUs detected, using MirroredStrategy with {len(gpus)} devices")
            self.strategy = tf.distribute.MirroredStrategy()
        
        print(f"Distribution strategy: {self.strategy.__class__.__name__}")
        print(f"Number of replicas: {self.strategy.num_replicas_in_sync}")
    
    def _setup_data_loader(self):
        """Setup data loader"""
        print("Setting up data loader...")
        self.data_loader = CIFAR10DataLoader(self.config)
        
        # Load datasets
        self.train_dataset, self.val_dataset, self.test_dataset = self.data_loader.load_datasets()
        
        # Get distributed datasets
        self.train_dist_dataset, self.val_dist_dataset, self.test_dist_dataset = \
            self.data_loader.get_distributed_datasets(self.strategy)
        
        # Print dataset information
        self.data_loader.print_dataset_info()
    
    def _setup_model(self):
        """Setup model within distributed strategy"""
        print("Setting up model...")
        
        with self.strategy.scope():
            self.model = create_model(self.config)
        
        # Print model information
        print_model_info(self.model)
    
    def _setup_callbacks(self):
        """Setup callbacks"""
        print("Setting up callbacks...")
        self.callbacks = create_callbacks(self.config)
        print_callback_info(self.callbacks)
    
    def train(self) -> keras.callbacks.History:
        """
        Train the model using distributed training
        
        Returns:
            Training history
        """
        print("\n" + "="*60)
        print("STARTING DISTRIBUTED TRAINING")
        print("="*60)
        
        # Record start time
        start_time = time.time()
        
        # Training configuration
        training_config = self.config.get_training_config()
        epochs = training_config["epochs"]
        
        print(f"Training for {epochs} epochs")
        print(f"Batch size: {training_config['batch_size']}")
        print(f"Learning rate: {training_config['learning_rate']}")
        print(f"Number of replicas: {self.strategy.num_replicas_in_sync}")
        
        # Train the model
        try:
            self.history = self.model.fit(
                self.train_dist_dataset,
                epochs=epochs,
                validation_data=self.val_dist_dataset,
                callbacks=self.callbacks,
                verbose=1
            )
            
            # Record training time
            self.training_time = time.time() - start_time
            
            # Update best metrics
            if self.history.history:
                self.best_accuracy = max(self.history.history.get('accuracy', [0]))
                self.best_val_accuracy = max(self.history.history.get('val_accuracy', [0]))
            
            print(f"\nTraining completed in {self.training_time:.2f} seconds")
            print(f"Best training accuracy: {self.best_accuracy:.4f}")
            print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise
        
        return self.history
    
    def evaluate(self, dataset_type: str = "test") -> Dict[str, float]:
        """
        Evaluate the model on specified dataset
        
        Args:
            dataset_type: Type of dataset to evaluate on ('test', 'val', 'train')
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating model on {dataset_type} dataset...")
        
        # Select dataset
        if dataset_type == "test":
            dataset = self.test_dist_dataset
        elif dataset_type == "val":
            dataset = self.val_dist_dataset
        elif dataset_type == "train":
            dataset = self.train_dist_dataset
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Evaluate model
        evaluation_results = self.model.evaluate(
            dataset,
            verbose=1,
            return_dict=True
        )
        
        # Print results
        print(f"\n{dataset_type.upper()} EVALUATION RESULTS:")
        print("="*40)
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value:.4f}")
        
        return evaluation_results
    
    def predict(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input images
        
        Args:
            images: Input images array
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Preprocess images if needed
        if images.dtype != np.float32:
            images = images.astype(np.float32)
        
        if images.max() > 1.0:
            images = images / 255.0
        
        # Make predictions
        predictions = self.model.predict(images, verbose=0)
        
        # Get predicted classes and probabilities
        predicted_classes = np.argmax(predictions, axis=1)
        probabilities = np.max(predictions, axis=1)
        
        return predicted_classes, probabilities
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        print(f"Saving model to {filepath}")
        self.model.save(filepath)
        print("Model saved successfully!")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        print(f"Loading model from {filepath}")
        self.model = keras.models.load_model(filepath)
        print("Model loaded successfully!")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training process
        
        Returns:
            Dictionary containing training summary
        """
        summary = {
            "training_time": self.training_time,
            "best_accuracy": self.best_accuracy,
            "best_val_accuracy": self.best_val_accuracy,
            "strategy": self.strategy.__class__.__name__,
            "num_replicas": self.strategy.num_replicas_in_sync,
            "model_type": self.config.get_model_config()["model_type"],
            "total_parameters": sum([tf.size(w).numpy() for w in self.model.trainable_weights]),
            "training_config": self.config.get_training_config(),
            "model_config": self.config.get_model_config()
        }
        
        if self.history and self.history.history:
            summary["final_epoch"] = len(self.history.history.get('loss', []))
            summary["final_loss"] = self.history.history.get('loss', [0])[-1]
            summary["final_val_loss"] = self.history.history.get('val_loss', [0])[-1]
        
        return summary
    
    def print_training_summary(self):
        """Print a detailed summary of the training process"""
        summary = self.get_training_summary()
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Training Time: {summary['training_time']:.2f} seconds")
        print(f"Best Training Accuracy: {summary['best_accuracy']:.4f}")
        print(f"Best Validation Accuracy: {summary['best_val_accuracy']:.4f}")
        print(f"Distribution Strategy: {summary['strategy']}")
        print(f"Number of Replicas: {summary['num_replicas']}")
        print(f"Model Type: {summary['model_type']}")
        print(f"Total Parameters: {summary['total_parameters']:,}")
        
        if 'final_epoch' in summary:
            print(f"Final Epoch: {summary['final_epoch']}")
            print(f"Final Loss: {summary['final_loss']:.4f}")
            print(f"Final Validation Loss: {summary['final_val_loss']:.4f}")
        
        print("="*60 + "\n")
    
    def save_training_results(self, save_dir: str = "results"):
        """
        Save training results and history
        
        Args:
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save training summary
        summary = self.get_training_summary()
        import json
        with open(os.path.join(save_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save callback history
        save_callback_history(self.callbacks, save_dir)
        
        # Save training history
        if self.history and self.history.history:
            with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
                json.dump(self.history.history, f, indent=2)
        
        print(f"Training results saved to {save_dir}/")

def create_trainer(config: Config) -> DistributedTrainer:
    """
    Create and return a distributed trainer
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized distributed trainer
    """
    return DistributedTrainer(config) 