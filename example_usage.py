"""
Example usage script for CIFAR-10 distributed training
Demonstrates different training configurations and model types
"""

import os
import sys
import time
import json
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import create_data_loader
from models import create_model, print_model_info
from distributed_trainer import create_trainer
from inference import create_inference_engine
from visualization import create_visualizer

def example_basic_training():
    """Example: Basic training with ResNet model"""
    print("\n" + "="*60)
    print("EXAMPLE 1: BASIC RESNET TRAINING")
    print("="*60)
    
    # Create configuration
    config = Config()
    config.MODEL_TYPE = "resnet"
    config.EPOCHS = 50  # Reduced for example
    config.BATCH_SIZE = 64
    config.LEARNING_RATE = 0.001
    
    # Create trainer and train
    trainer = create_trainer(config)
    history = trainer.train()
    
    # Evaluate model
    test_results = trainer.evaluate("test")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    
    # Save model
    trainer.save_model("models/resnet_basic.h5")
    
    return trainer, history, test_results

def example_custom_cnn_training():
    """Example: Custom CNN training with different parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 2: CUSTOM CNN TRAINING")
    print("="*60)
    
    # Create configuration
    config = Config()
    config.MODEL_TYPE = "cnn"
    config.EPOCHS = 30
    config.BATCH_SIZE = 32
    config.LEARNING_RATE = 0.0005
    config.DROPOUT_RATE = 0.5
    config.EARLY_STOPPING_PATIENCE = 10
    
    # Create trainer and train
    trainer = create_trainer(config)
    history = trainer.train()
    
    # Evaluate model
    test_results = trainer.evaluate("test")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    
    # Save model
    trainer.save_model("models/cnn_custom.h5")
    
    return trainer, history, test_results

def example_efficientnet_training():
    """Example: EfficientNet training with transfer learning"""
    print("\n" + "="*60)
    print("EXAMPLE 3: EFFICIENTNET TRAINING")
    print("="*60)
    
    # Create configuration
    config = Config()
    config.MODEL_TYPE = "efficientnet"
    config.EPOCHS = 20  # Fewer epochs for transfer learning
    config.BATCH_SIZE = 32
    config.LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning
    config.EARLY_STOPPING_PATIENCE = 8
    
    # Create trainer and train
    trainer = create_trainer(config)
    history = trainer.train()
    
    # Evaluate model
    test_results = trainer.evaluate("test")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    
    # Save model
    trainer.save_model("models/efficientnet_transfer.h5")
    
    return trainer, history, test_results

def example_inference_only():
    """Example: Load trained model and run inference"""
    print("\n" + "="*60)
    print("EXAMPLE 4: INFERENCE ONLY")
    print("="*60)
    
    # Create configuration
    config = Config()
    
    # Check if model exists
    model_path = "models/resnet_basic.h5"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train a model first.")
        return None
    
    # Create inference engine
    inference_engine = create_inference_engine(config, model_path)
    
    # Load test data
    data_loader = create_data_loader(config)
    _, _, test_dataset = data_loader.load_datasets()
    
    # Get sample data
    images, labels = data_loader.get_sample_batch("test")
    true_classes = tf.argmax(labels, axis=1).numpy()
    
    # Run inference
    predicted_classes, confidences, _ = inference_engine.predict_batch(images.numpy())
    
    # Evaluate predictions
    evaluation_results = inference_engine.evaluate_predictions(
        true_classes, predicted_classes, confidences
    )
    
    # Print results
    inference_engine.print_inference_summary(evaluation_results)
    
    # Visualize predictions
    inference_engine.visualize_predictions(
        images.numpy(), predicted_classes, confidences, true_classes,
        save_path="results/example_inference_predictions.png"
    )
    
    return inference_engine, evaluation_results

def example_comparison_training():
    """Example: Compare different model architectures"""
    print("\n" + "="*60)
    print("EXAMPLE 5: MODEL COMPARISON")
    print("="*60)
    
    results = {}
    
    # Test different model types
    model_types = ["resnet", "cnn", "efficientnet"]
    
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model...")
        
        # Create configuration
        config = Config()
        config.MODEL_TYPE = model_type
        config.EPOCHS = 10  # Quick training for comparison
        config.BATCH_SIZE = 64
        config.LEARNING_RATE = 0.001
        
        # Create trainer and train
        trainer = create_trainer(config)
        history = trainer.train()
        
        # Evaluate model
        test_results = trainer.evaluate("test")
        
        # Store results
        results[model_type] = {
            "test_accuracy": test_results['accuracy'],
            "training_time": trainer.training_time,
            "final_loss": history.history.get('loss', [0])[-1] if history else 0,
            "final_val_loss": history.history.get('val_loss', [0])[-1] if history else 0
        }
        
        # Save model
        trainer.save_model(f"models/{model_type}_comparison.h5")
    
    # Print comparison results
    print("\n" + "="*40)
    print("MODEL COMPARISON RESULTS")
    print("="*40)
    for model_type, result in results.items():
        print(f"{model_type.upper()}:")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Training Time: {result['training_time']:.2f} seconds")
        print(f"  Final Loss: {result['final_loss']:.4f}")
        print(f"  Final Val Loss: {result['final_val_loss']:.4f}")
        print()
    
    return results

def example_visualization():
    """Example: Create comprehensive visualizations"""
    print("\n" + "="*60)
    print("EXAMPLE 6: COMPREHENSIVE VISUALIZATION")
    print("="*60)
    
    # Create configuration
    config = Config()
    
    # Create visualizer
    visualizer = create_visualizer(config)
    
    # Load training history (if available)
    history_file = "results/training_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Create training history plot
        visualizer.plot_training_history(
            history,
            save_path="results/example_training_history.png"
        )
        
        print("Training history visualization created!")
    else:
        print("No training history found. Run training first.")
    
    # Load evaluation results (if available)
    eval_file = "results/inference_results.json"
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            eval_results = json.load(f)
        
        # Create confusion matrix
        if 'confusion_matrix' in eval_results:
            cm = np.array(eval_results['confusion_matrix'])
            visualizer.plot_confusion_matrix(
                cm,
                save_path="results/example_confusion_matrix.png"
            )
        
        # Create class accuracy plot
        if 'per_class_accuracy' in eval_results:
            visualizer.plot_class_accuracy(
                eval_results['per_class_accuracy'],
                save_path="results/example_class_accuracy.png"
            )
        
        print("Evaluation visualizations created!")
    else:
        print("No evaluation results found. Run inference first.")

def example_data_exploration():
    """Example: Explore the CIFAR-10 dataset"""
    print("\n" + "="*60)
    print("EXAMPLE 7: DATA EXPLORATION")
    print("="*60)
    
    # Create configuration
    config = Config()
    
    # Create data loader
    data_loader = create_data_loader(config)
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = data_loader.load_datasets()
    
    # Print dataset information
    data_loader.print_dataset_info()
    
    # Visualize sample images
    data_loader.visualize_samples(num_samples=16, dataset_type="train")
    data_loader.visualize_samples(num_samples=16, dataset_type="test")
    
    print("Data exploration completed!")

def main():
    """Run all examples"""
    print("CIFAR-10 DISTRIBUTED TRAINING EXAMPLES")
    print("="*60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Run examples
    try:
        # Example 1: Basic training
        example_basic_training()
        
        # Example 2: Custom CNN
        example_custom_cnn_training()
        
        # Example 3: EfficientNet
        example_efficientnet_training()
        
        # Example 4: Inference only
        example_inference_only()
        
        # Example 5: Model comparison
        example_comparison_training()
        
        # Example 6: Visualization
        example_visualization()
        
        # Example 7: Data exploration
        example_data_exploration()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check the 'results/' and 'models/' directories for outputs.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 