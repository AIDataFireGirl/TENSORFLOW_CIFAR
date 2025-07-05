"""
Model architectures for CIFAR-10 classification
Includes ResNet, custom CNN, and EfficientNet variants with distributed training support
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Any, Optional, Tuple
import numpy as np
from config import Config

class ResidualBlock(layers.Layer):
    """Residual block for ResNet architecture"""
    
    def __init__(self, filters: int, kernel_size: int = 3, strides: int = 1, **kwargs):
        """
        Initialize residual block
        
        Args:
            filters: Number of filters in the convolutional layers
            kernel_size: Size of the convolutional kernel
            strides: Stride for the first convolution
            **kwargs: Additional arguments for the layer
        """
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        
        # First convolutional block
        self.conv1 = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=False
        )
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.ReLU()
        
        # Second convolutional block
        self.conv2 = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            use_bias=False
        )
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut connection
        self.shortcut = None
        if strides != 1 or filters != filters:
            self.shortcut = layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=strides,
                padding='same',
                use_bias=False
            )
            self.shortcut_bn = layers.BatchNormalization()
        
        self.add_layer = layers.Add()
        self.activation2 = layers.ReLU()
    
    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward pass through the residual block
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut path
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
        
        # Add main and shortcut paths
        x = self.add_layer([x, shortcut])
        x = self.activation2(x)
        
        return x

class ResNetModel(keras.Model):
    """ResNet model for CIFAR-10 classification"""
    
    def __init__(self, config: Config, **kwargs):
        """
        Initialize ResNet model
        
        Args:
            config: Configuration object
            **kwargs: Additional arguments for the model
        """
        super(ResNetModel, self).__init__(**kwargs)
        self.config = config
        
        # Model parameters
        self.input_shape = config.get_model_config()["input_shape"]
        self.num_classes = config.get_model_config()["num_classes"]
        self.dropout_rate = config.get_model_config()["dropout_rate"]
        self.regularization_factor = config.get_model_config()["regularization_factor"]
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        """Build the ResNet architecture"""
        # Input layer
        self.input_layer = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        self.conv1 = layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.regularization_factor)
        )
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.ReLU()
        
        # Residual blocks
        self.residual_blocks = []
        
        # First group: 64 filters
        for _ in range(2):
            self.residual_blocks.append(ResidualBlock(64))
        
        # Second group: 128 filters
        self.residual_blocks.append(ResidualBlock(128, strides=2))
        for _ in range(1):
            self.residual_blocks.append(ResidualBlock(128))
        
        # Third group: 256 filters
        self.residual_blocks.append(ResidualBlock(256, strides=2))
        for _ in range(1):
            self.residual_blocks.append(ResidualBlock(256))
        
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling2D()
        
        # Dropout for regularization
        self.dropout = layers.Dropout(self.dropout_rate)
        
        # Output layer
        self.output_layer = layers.Dense(
            self.num_classes,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l2(self.regularization_factor)
        )
    
    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward pass through the model
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x, training=training)
        
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        x = self.output_layer(x)
        
        return x

class CustomCNNModel(keras.Model):
    """Custom CNN model for CIFAR-10 classification"""
    
    def __init__(self, config: Config, **kwargs):
        """
        Initialize custom CNN model
        
        Args:
            config: Configuration object
            **kwargs: Additional arguments for the model
        """
        super(CustomCNNModel, self).__init__(**kwargs)
        self.config = config
        
        # Model parameters
        self.input_shape = config.get_model_config()["input_shape"]
        self.num_classes = config.get_model_config()["num_classes"]
        self.dropout_rate = config.get_model_config()["dropout_rate"]
        self.regularization_factor = config.get_model_config()["regularization_factor"]
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        """Build the custom CNN architecture"""
        # Input layer
        self.input_layer = layers.Input(shape=self.input_shape)
        
        # First convolutional block
        self.conv1 = layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.regularization_factor)
        )
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(pool_size=2)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        
        # Second convolutional block
        self.conv2 = layers.Conv2D(
            filters=64,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.regularization_factor)
        )
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(pool_size=2)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        
        # Third convolutional block
        self.conv3 = layers.Conv2D(
            filters=128,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.regularization_factor)
        )
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(pool_size=2)
        self.dropout3 = layers.Dropout(self.dropout_rate)
        
        # Fourth convolutional block
        self.conv4 = layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.regularization_factor)
        )
        self.bn4 = layers.BatchNormalization()
        self.pool4 = layers.MaxPooling2D(pool_size=2)
        self.dropout4 = layers.Dropout(self.dropout_rate)
        
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling2D()
        
        # Dense layers
        self.dense1 = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.regularization_factor)
        )
        self.bn_dense1 = layers.BatchNormalization()
        self.dropout_dense1 = layers.Dropout(self.dropout_rate)
        
        self.dense2 = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.regularization_factor)
        )
        self.bn_dense2 = layers.BatchNormalization()
        self.dropout_dense2 = layers.Dropout(self.dropout_rate)
        
        # Output layer
        self.output_layer = layers.Dense(
            self.num_classes,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l2(self.regularization_factor)
        )
    
    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward pass through the model
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        x = self.dropout3(x, training=training)
        
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.pool4(x)
        x = self.dropout4(x, training=training)
        
        x = self.global_pool(x)
        
        x = self.dense1(x)
        x = self.bn_dense1(x, training=training)
        x = self.dropout_dense1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn_dense2(x, training=training)
        x = self.dropout_dense2(x, training=training)
        
        x = self.output_layer(x)
        
        return x

class EfficientNetModel(keras.Model):
    """EfficientNet model for CIFAR-10 classification"""
    
    def __init__(self, config: Config, **kwargs):
        """
        Initialize EfficientNet model
        
        Args:
            config: Configuration object
            **kwargs: Additional arguments for the model
        """
        super(EfficientNetModel, self).__init__(**kwargs)
        self.config = config
        
        # Model parameters
        self.input_shape = config.get_model_config()["input_shape"]
        self.num_classes = config.get_model_config()["num_classes"]
        self.dropout_rate = config.get_model_config()["dropout_rate"]
        self.regularization_factor = config.get_model_config()["regularization_factor"]
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        """Build the EfficientNet architecture"""
        # Use pre-trained EfficientNetB0 as base
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the model
        self.base_model = base_model
        
        # Add custom layers on top
        self.dropout = layers.Dropout(self.dropout_rate)
        self.output_layer = layers.Dense(
            self.num_classes,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l2(self.regularization_factor)
        )
    
    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward pass through the model
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        x = self.base_model(inputs, training=training)
        x = self.dropout(x, training=training)
        x = self.output_layer(x)
        
        return x

def create_model(config: Config) -> keras.Model:
    """
    Create a model based on the configuration
    
    Args:
        config: Configuration object
        
    Returns:
        Compiled model
    """
    model_type = config.get_model_config()["model_type"]
    
    if model_type == "resnet":
        model = ResNetModel(config)
    elif model_type == "cnn":
        model = CustomCNNModel(config)
    elif model_type == "efficientnet":
        model = EfficientNetModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            beta_1=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model

def get_model_summary(model: keras.Model) -> str:
    """
    Get a string representation of the model summary
    
    Args:
        model: Keras model
        
    Returns:
        Model summary as string
    """
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return '\n'.join(summary_list)

def count_model_parameters(model: keras.Model) -> Dict[str, int]:
    """
    Count trainable and non-trainable parameters in the model
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    return {
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'total': total_params
    }

def print_model_info(model: keras.Model):
    """Print detailed information about the model"""
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    
    # Print model summary
    print("\nModel Architecture:")
    print(get_model_summary(model))
    
    # Print parameter counts
    param_counts = count_model_parameters(model)
    print(f"\nParameters:")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Non-trainable: {param_counts['non_trainable']:,}")
    print(f"  Total: {param_counts['total']:,}")
    
    print("="*50 + "\n") 