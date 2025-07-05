"""
Callbacks module for distributed training
Includes early stopping, model checkpointing, learning rate scheduling, and TensorBoard logging
"""

import tensorflow as tf
from tensorflow import keras
import os
import datetime
from typing import List, Dict, Any, Optional
from config import Config

class CustomEarlyStopping(keras.callbacks.EarlyStopping):
    """Enhanced early stopping callback with detailed logging"""
    
    def __init__(self, config: Config, **kwargs):
        """
        Initialize custom early stopping
        
        Args:
            config: Configuration object
            **kwargs: Additional arguments for early stopping
        """
        super().__init__(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            **kwargs
        )
        self.config = config
        self.best_epoch = 0
        self.best_val_loss = float('inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """
        Called at the end of each epoch
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        if logs is None:
            logs = {}
        
        current_val_loss = logs.get('val_loss', float('inf'))
        
        # Update best values
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch
            print(f"New best validation loss: {current_val_loss:.4f} at epoch {epoch}")
        
        # Call parent method
        super().on_epoch_end(epoch, logs)
    
    def on_train_end(self, logs: Optional[Dict[str, float]] = None):
        """
        Called at the end of training
        
        Args:
            logs: Dictionary of metrics
        """
        print(f"\nEarly stopping summary:")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Stopped at epoch: {self.best_epoch + self.patience}")
        
        super().on_train_end(logs)

class CustomModelCheckpoint(keras.callbacks.ModelCheckpoint):
    """Enhanced model checkpointing with multiple save strategies"""
    
    def __init__(self, config: Config, **kwargs):
        """
        Initialize custom model checkpointing
        
        Args:
            config: Configuration object
            **kwargs: Additional arguments for model checkpointing
        """
        # Create checkpoint directory
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
        # Set up file paths
        best_model_path = config.BEST_MODEL_PATH
        latest_model_path = config.LATEST_MODEL_PATH
        
        super().__init__(
            filepath=best_model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            **kwargs
        )
        
        self.config = config
        self.latest_model_path = latest_model_path
        self.best_model_path = best_model_path
        self.checkpoint_history = []
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """
        Called at the end of each epoch
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        if logs is None:
            logs = {}
        
        # Save latest model
        self.model.save(self.latest_model_path)
        
        # Record checkpoint info
        checkpoint_info = {
            'epoch': epoch,
            'val_loss': logs.get('val_loss', 0.0),
            'val_accuracy': logs.get('val_accuracy', 0.0),
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.checkpoint_history.append(checkpoint_info)
        
        # Call parent method for best model saving
        super().on_epoch_end(epoch, logs)
    
    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of checkpoints
        
        Returns:
            List of checkpoint information dictionaries
        """
        return self.checkpoint_history.copy()

class CustomReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau):
    """Enhanced learning rate reduction callback with detailed logging"""
    
    def __init__(self, config: Config, **kwargs):
        """
        Initialize custom learning rate reduction
        
        Args:
            config: Configuration object
            **kwargs: Additional arguments for learning rate reduction
        """
        super().__init__(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
            **kwargs
        )
        self.config = config
        self.lr_history = []
        self.reduction_count = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """
        Called at the end of each epoch
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        if logs is None:
            logs = {}
        
        # Record current learning rate
        current_lr = self.model.optimizer.learning_rate.numpy()
        self.lr_history.append({
            'epoch': epoch,
            'learning_rate': current_lr,
            'val_loss': logs.get('val_loss', 0.0)
        })
        
        # Call parent method
        super().on_epoch_end(epoch, logs)
    
    def on_reduce_lr(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """
        Called when learning rate is reduced
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        self.reduction_count += 1
        new_lr = self.model.optimizer.learning_rate.numpy()
        print(f"\nLearning rate reduced at epoch {epoch}")
        print(f"  New learning rate: {new_lr:.2e}")
        print(f"  Total reductions: {self.reduction_count}")
    
    def get_lr_history(self) -> List[Dict[str, Any]]:
        """
        Get the learning rate history
        
        Returns:
            List of learning rate information dictionaries
        """
        return self.lr_history.copy()

class CustomTensorBoard(keras.callbacks.TensorBoard):
    """Enhanced TensorBoard callback with additional logging"""
    
    def __init__(self, config: Config, **kwargs):
        """
        Initialize custom TensorBoard callback
        
        Args:
            config: Configuration object
            **kwargs: Additional arguments for TensorBoard
        """
        # Create log directory
        os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)
        
        # Set up log directory with timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, current_time)
        
        super().__init__(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            **kwargs
        )
        
        self.config = config
        self.log_dir = log_dir
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """
        Called at the end of each epoch
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        if logs is None:
            logs = {}
        
        # Log additional metrics
        with self.writer.as_default():
            # Log learning rate
            lr = self.model.optimizer.learning_rate.numpy()
            tf.summary.scalar('learning_rate', lr, step=epoch)
            
            # Log parameter statistics
            for layer in self.model.layers:
                if hasattr(layer, 'kernel'):
                    weights = layer.kernel
                    tf.summary.histogram(f'{layer.name}/kernel', weights, step=epoch)
                    tf.summary.scalar(f'{layer.name}/kernel_mean', tf.reduce_mean(weights), step=epoch)
                    tf.summary.scalar(f'{layer.name}/kernel_std', tf.math.reduce_std(weights), step=epoch)
        
        # Call parent method
        super().on_epoch_end(epoch, logs)

class CustomCSVLogger(keras.callbacks.CSVLogger):
    """Enhanced CSV logger with additional metrics"""
    
    def __init__(self, config: Config, **kwargs):
        """
        Initialize custom CSV logger
        
        Args:
            config: Configuration object
            **kwargs: Additional arguments for CSV logger
        """
        # Create log directory
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # Set up log file path
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(config.LOG_DIR, f"training_log_{current_time}.csv")
        
        super().__init__(
            filename=log_file,
            separator=',',
            append=False,
            **kwargs
        )
        
        self.config = config
        self.log_file = log_file
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """
        Called at the end of each epoch
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        if logs is None:
            logs = {}
        
        # Add additional metrics
        logs['learning_rate'] = self.model.optimizer.learning_rate.numpy()
        logs['epoch'] = epoch
        
        # Call parent method
        super().on_epoch_end(epoch, logs)

def create_callbacks(config: Config) -> List[keras.callbacks.Callback]:
    """
    Create a list of callbacks for training
    
    Args:
        config: Configuration object
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Early stopping
    early_stopping = CustomEarlyStopping(config)
    callbacks.append(early_stopping)
    
    # Model checkpointing
    model_checkpoint = CustomModelCheckpoint(config)
    callbacks.append(model_checkpoint)
    
    # Learning rate reduction
    reduce_lr = CustomReduceLROnPlateau(config)
    callbacks.append(reduce_lr)
    
    # TensorBoard logging
    tensorboard = CustomTensorBoard(config)
    callbacks.append(tensorboard)
    
    # CSV logging
    csv_logger = CustomCSVLogger(config)
    callbacks.append(csv_logger)
    
    return callbacks

def print_callback_info(callbacks: List[keras.callbacks.Callback]):
    """Print information about the configured callbacks"""
    print("\n" + "="*50)
    print("CALLBACKS CONFIGURATION")
    print("="*50)
    
    for i, callback in enumerate(callbacks, 1):
        print(f"{i}. {callback.__class__.__name__}")
        
        if isinstance(callback, CustomEarlyStopping):
            print(f"   - Monitor: {callback.monitor}")
            print(f"   - Patience: {callback.patience}")
            print(f"   - Restore best weights: {callback.restore_best_weights}")
        
        elif isinstance(callback, CustomModelCheckpoint):
            print(f"   - Filepath: {callback.filepath}")
            print(f"   - Monitor: {callback.monitor}")
            print(f"   - Save best only: {callback.save_best_only}")
        
        elif isinstance(callback, CustomReduceLROnPlateau):
            print(f"   - Monitor: {callback.monitor}")
            print(f"   - Factor: {callback.factor}")
            print(f"   - Patience: {callback.patience}")
            print(f"   - Min LR: {callback.min_lr}")
        
        elif isinstance(callback, CustomTensorBoard):
            print(f"   - Log directory: {callback.log_dir}")
            print(f"   - Histogram freq: {callback.histogram_freq}")
            print(f"   - Write graph: {callback.write_graph}")
        
        elif isinstance(callback, CustomCSVLogger):
            print(f"   - Log file: {callback.log_file}")
            print(f"   - Separator: {callback.separator}")
    
    print("="*50 + "\n")

def save_callback_history(callbacks: List[keras.callbacks.Callback], save_dir: str = "results"):
    """
    Save callback history to files
    
    Args:
        callbacks: List of callbacks
        save_dir: Directory to save history files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for callback in callbacks:
        if hasattr(callback, 'get_checkpoint_history'):
            history = callback.get_checkpoint_history()
            if history:
                import json
                with open(os.path.join(save_dir, 'checkpoint_history.json'), 'w') as f:
                    json.dump(history, f, indent=2)
        
        elif hasattr(callback, 'get_lr_history'):
            history = callback.get_lr_history()
            if history:
                import json
                with open(os.path.join(save_dir, 'lr_history.json'), 'w') as f:
                    json.dump(history, f, indent=2) 