"""
Data loader module for CIFAR-10 dataset
Handles data loading, preprocessing, augmentation, and distributed training support
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config, CIFAR10_CLASSES, CLASS_COLORS

class CIFAR10DataLoader:
    """Data loader class for CIFAR-10 dataset with distributed training support"""
    
    def __init__(self, config: Config):
        """
        Initialize the data loader
        
        Args:
            config: Configuration object containing dataset parameters
        """
        self.config = config
        self.dataset_name = config.DATASET_NAME
        self.image_size = config.IMAGE_SIZE
        self.num_classes = config.NUM_CLASSES
        self.batch_size = config.BATCH_SIZE
        self.validation_split = config.VALIDATION_SPLIT
        self.test_split = config.TEST_SPLIT
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Data augmentation layers
        self.augmentation_layers = self._create_augmentation_layers()
        
    def _create_augmentation_layers(self) -> tf.keras.Sequential:
        """Create data augmentation layers for training"""
        return tf.keras.Sequential([
            # Random rotation
            tf.keras.layers.RandomRotation(
                factor=self.config.ROTATION_RANGE / 360.0,
                fill_mode='nearest'
            ),
            # Random width and height shifts
            tf.keras.layers.RandomTranslation(
                height_factor=self.config.HEIGHT_SHIFT_RANGE,
                width_factor=self.config.WIDTH_SHIFT_RANGE,
                fill_mode='nearest'
            ),
            # Random zoom
            tf.keras.layers.RandomZoom(
                height_factor=self.config.ZOOM_RANGE,
                width_factor=self.config.ZOOM_RANGE,
                fill_mode='nearest'
            ),
            # Random horizontal flip
            tf.keras.layers.RandomFlip("horizontal") if self.config.HORIZONTAL_FLIP else tf.keras.layers.Layer(),
        ])
    
    def _preprocess_image(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocess image and label for training
        
        Args:
            image: Input image tensor
            label: Input label tensor
            
        Returns:
            Tuple of preprocessed image and one-hot encoded label
        """
        # Convert image to float32 and normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # Resize image if needed (CIFAR-10 is already 32x32)
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = tf.image.resize(image, (self.image_size, self.image_size))
        
        # One-hot encode the label
        label = tf.one_hot(label, self.num_classes)
        
        return image, label
    
    def _augment_image(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image tensor
            label: Input label tensor
            
        Returns:
            Tuple of augmented image and label
        """
        # Apply augmentation layers
        image = self.augmentation_layers(image, training=True)
        return image, label
    
    def load_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load CIFAR-10 datasets with train/validation/test splits
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        print("Loading CIFAR-10 dataset...")
        
        # Load the full dataset
        dataset, info = tfds.load(
            self.dataset_name,
            as_supervised=True,
            with_info=True
        )
        
        # Get dataset sizes
        total_size = info.splits['train'].num_examples
        train_size = int(total_size * (1 - self.validation_split - self.test_split))
        val_size = int(total_size * self.validation_split)
        test_size = total_size - train_size - val_size
        
        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # Split the dataset
        train_dataset = dataset['train'].take(train_size)
        remaining_dataset = dataset['train'].skip(train_size)
        
        val_dataset = remaining_dataset.take(val_size)
        test_dataset = remaining_dataset.skip(val_size).take(test_size)
        
        # Apply preprocessing and augmentation
        self.train_dataset = self._prepare_dataset(
            train_dataset, 
            augment=True, 
            shuffle=True
        )
        
        self.val_dataset = self._prepare_dataset(
            val_dataset, 
            augment=False, 
            shuffle=False
        )
        
        self.test_dataset = self._prepare_dataset(
            test_dataset, 
            augment=False, 
            shuffle=False
        )
        
        print("Dataset loading completed!")
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def _prepare_dataset(
        self, 
        dataset: tf.data.Dataset, 
        augment: bool = False, 
        shuffle: bool = False
    ) -> tf.data.Dataset:
        """
        Prepare dataset with preprocessing and optional augmentation
        
        Args:
            dataset: Raw dataset
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle the dataset
            
        Returns:
            Prepared dataset
        """
        # Apply preprocessing
        dataset = dataset.map(
            self._preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation if requested
        if augment:
            dataset = dataset.map(
                self._augment_image,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        # Batch the dataset
        dataset = dataset.batch(self.batch_size)
        
        # Prefetch for better performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_distributed_datasets(self, strategy: tf.distribute.Strategy) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Get datasets distributed across multiple GPUs/TPUs
        
        Args:
            strategy: TensorFlow distribution strategy
            
        Returns:
            Tuple of distributed datasets
        """
        if self.train_dataset is None:
            self.load_datasets()
        
        # Distribute datasets across devices
        train_dist_dataset = strategy.experimental_distribute_dataset(self.train_dataset)
        val_dist_dataset = strategy.experimental_distribute_dataset(self.val_dataset)
        test_dist_dataset = strategy.experimental_distribute_dataset(self.test_dataset)
        
        return train_dist_dataset, val_dist_dataset, test_dist_dataset
    
    def get_sample_batch(self, dataset_type: str = "train") -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get a sample batch for visualization
        
        Args:
            dataset_type: Type of dataset ('train', 'val', 'test')
            
        Returns:
            Tuple of (images, labels)
        """
        if dataset_type == "train" and self.train_dataset is not None:
            dataset = self.train_dataset
        elif dataset_type == "val" and self.val_dataset is not None:
            dataset = self.val_dataset
        elif dataset_type == "test" and self.test_dataset is not None:
            dataset = self.test_dataset
        else:
            raise ValueError(f"Dataset {dataset_type} not loaded")
        
        # Get first batch
        for images, labels in dataset.take(1):
            return images, labels
    
    def visualize_samples(self, num_samples: int = 16, dataset_type: str = "train"):
        """
        Visualize sample images from the dataset
        
        Args:
            num_samples: Number of samples to visualize
            dataset_type: Type of dataset to visualize
        """
        images, labels = self.get_sample_batch(dataset_type)
        
        # Convert one-hot labels back to class indices
        label_indices = tf.argmax(labels, axis=1)
        
        # Create subplot
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            # Denormalize image
            img = images[i] * 255.0
            img = tf.cast(img, tf.uint8)
            
            # Display image
            axes[i].imshow(img)
            axes[i].set_title(f"{CIFAR10_CLASSES[label_indices[i]]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"results/sample_images_{dataset_type}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded datasets
        
        Returns:
            Dictionary containing dataset information
        """
        info = {
            "dataset_name": self.dataset_name,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "test_split": self.test_split
        }
        
        if self.train_dataset is not None:
            # Count samples in each dataset
            train_steps = len(list(self.train_dataset))
            val_steps = len(list(self.val_dataset))
            test_steps = len(list(self.test_dataset))
            
            info.update({
                "train_samples": train_steps * self.batch_size,
                "val_samples": val_steps * self.batch_size,
                "test_samples": test_steps * self.batch_size,
                "train_batches": train_steps,
                "val_batches": val_steps,
                "test_batches": test_steps
            })
        
        return info
    
    def print_dataset_info(self):
        """Print detailed information about the datasets"""
        info = self.get_dataset_info()
        
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"Dataset: {info['dataset_name']}")
        print(f"Image Size: {info['image_size']}x{info['image_size']}")
        print(f"Number of Classes: {info['num_classes']}")
        print(f"Batch Size: {info['batch_size']}")
        print(f"Validation Split: {info['validation_split']:.1%}")
        print(f"Test Split: {info['test_split']:.1%}")
        
        if 'train_samples' in info:
            print(f"\nSamples:")
            print(f"  Train: {info['train_samples']:,}")
            print(f"  Validation: {info['val_samples']:,}")
            print(f"  Test: {info['test_samples']:,}")
            print(f"\nBatches:")
            print(f"  Train: {info['train_batches']}")
            print(f"  Validation: {info['val_batches']}")
            print(f"  Test: {info['test_batches']}")
        
        print("="*50 + "\n")

# Utility function for creating data loader
def create_data_loader(config: Config) -> CIFAR10DataLoader:
    """
    Create and return a CIFAR-10 data loader
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized data loader
    """
    return CIFAR10DataLoader(config) 