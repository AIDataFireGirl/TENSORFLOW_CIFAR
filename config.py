"""
Configuration file for CIFAR-10 distributed training and inference
Contains all hyperparameters, model settings, and training configurations
"""

import os
from typing import Dict, Any

class Config:
    """Centralized configuration for CIFAR-10 distributed training"""
    
    # Dataset Configuration
    DATASET_NAME = "cifar10"
    IMAGE_SIZE = 32
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    
    # Training Configuration
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    
    # Distributed Training Configuration
    STRATEGY = "mirrored"  # Options: "mirrored", "multi_worker_mirrored", "tpu"
    NUM_GPUS = 2  # Number of GPUs to use for distributed training
    
    # Model Configuration
    MODEL_TYPE = "resnet"  # Options: "resnet", "cnn", "efficientnet"
    DROPOUT_RATE = 0.3
    REGULARIZATION_FACTOR = 1e-4
    
    # Callbacks Configuration
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 10
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7
    
    # Checkpoint Configuration
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = "checkpoints/best_model.h5"
    LATEST_MODEL_PATH = "checkpoints/latest_model.h5"
    
    # Logging Configuration
    LOG_DIR = "logs"
    TENSORBOARD_LOG_DIR = "logs/tensorboard"
    
    # Data Augmentation Configuration
    ROTATION_RANGE = 15
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    HORIZONTAL_FLIP = True
    ZOOM_RANGE = 0.1
    
    # Validation Configuration
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Inference Configuration
    INFERENCE_BATCH_SIZE = 32
    CONFIDENCE_THRESHOLD = 0.5
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            "input_shape": (cls.IMAGE_SIZE, cls.IMAGE_SIZE, cls.NUM_CHANNELS),
            "num_classes": cls.NUM_CLASSES,
            "dropout_rate": cls.DROPOUT_RATE,
            "regularization_factor": cls.REGULARIZATION_FACTOR,
            "model_type": cls.MODEL_TYPE
        }
    
    @classmethod
    def get_training_config(cls) -> Dict[str, Any]:
        """Get training-specific configuration"""
        return {
            "batch_size": cls.BATCH_SIZE,
            "epochs": cls.EPOCHS,
            "learning_rate": cls.LEARNING_RATE,
            "momentum": cls.MOMENTUM,
            "weight_decay": cls.WEIGHT_DECAY,
            "validation_split": cls.VALIDATION_SPLIT
        }
    
    @classmethod
    def get_distributed_config(cls) -> Dict[str, Any]:
        """Get distributed training configuration"""
        return {
            "strategy": cls.STRATEGY,
            "num_gpus": cls.NUM_GPUS
        }
    
    @classmethod
    def get_callback_config(cls) -> Dict[str, Any]:
        """Get callback configuration"""
        return {
            "early_stopping_patience": cls.EARLY_STOPPING_PATIENCE,
            "reduce_lr_patience": cls.REDUCE_LR_PATIENCE,
            "reduce_lr_factor": cls.REDUCE_LR_FACTOR,
            "min_lr": cls.MIN_LR,
            "checkpoint_dir": cls.CHECKPOINT_DIR,
            "best_model_path": cls.BEST_MODEL_PATH,
            "latest_model_path": cls.LATEST_MODEL_PATH,
            "log_dir": cls.LOG_DIR,
            "tensorboard_log_dir": cls.TENSORBOARD_LOG_DIR
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for the project"""
        directories = [
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.TENSORBOARD_LOG_DIR,
            "models",
            "results",
            "data"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Color mapping for visualization
CLASS_COLORS = {
    'airplane': '#FF6B6B',
    'automobile': '#4ECDC4',
    'bird': '#45B7D1',
    'cat': '#96CEB4',
    'deer': '#FFEAA7',
    'dog': '#DDA0DD',
    'frog': '#98D8C8',
    'horse': '#F7DC6F',
    'ship': '#BB8FCE',
    'truck': '#85C1E9'
} 