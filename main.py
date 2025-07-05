"""
Main training script for CIFAR-10 distributed training and inference
Orchestrates the entire pipeline including training, evaluation, and visualization
"""

import os
import sys
import argparse
import time
import datetime
from typing import Dict, Any, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import create_data_loader
from models import create_model, print_model_info
from distributed_trainer import create_trainer
from inference import create_inference_engine
from visualization import create_visualizer

def setup_environment():
    """Setup the training environment"""
    print("="*60)
    print("CIFAR-10 DISTRIBUTED TRAINING SETUP")
    print("="*60)
    
    # Create necessary directories
    Config.create_directories()
    
    # Print system information
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    print(f"Available GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    
    # Set memory growth for GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU memory growth setup failed: {e}")

def train_model(config: Config, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Train the model using distributed training
    
    Args:
        config: Configuration object
        model_path: Path to load existing model (optional)
        
    Returns:
        Dictionary containing training results
    """
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Load existing model if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        trainer.load_model(model_path)
    
    # Train the model
    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_results = trainer.evaluate("test")
    val_results = trainer.evaluate("val")
    
    # Save the model
    trainer.save_model(config.BEST_MODEL_PATH)
    
    # Save training results
    trainer.save_training_results()
    
    # Print training summary
    trainer.print_training_summary()
    
    # Prepare results
    results = {
        "training_time": training_time,
        "history": history.history if history else {},
        "test_results": test_results,
        "val_results": val_results,
        "model_path": config.BEST_MODEL_PATH
    }
    
    return results

def run_inference(config: Config, model_path: str, test_data: bool = True) -> Dict[str, Any]:
    """
    Run inference on the trained model
    
    Args:
        config: Configuration object
        model_path: Path to the trained model
        test_data: Whether to run inference on test data
        
    Returns:
        Dictionary containing inference results
    """
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    # Create inference engine
    inference_engine = create_inference_engine(config, model_path)
    
    # Load test data for inference
    if test_data:
        data_loader = create_data_loader(config)
        _, _, test_dataset = data_loader.load_datasets()
        
        # Get sample data for inference
        images, labels = data_loader.get_sample_batch("test")
        
        # Convert one-hot labels to class indices
        true_classes = tf.argmax(labels, axis=1).numpy()
        
        # Run inference
        predicted_classes, confidences, all_probabilities = inference_engine.predict_batch(images.numpy())
        
        # Evaluate predictions
        evaluation_results = inference_engine.evaluate_predictions(
            true_classes, predicted_classes, confidences
        )
        
        # Print results
        inference_engine.print_inference_summary(evaluation_results)
        
        # Visualize predictions
        inference_engine.visualize_predictions(
            images.numpy(), predicted_classes, confidences, true_classes,
            save_path="results/prediction_samples.png"
        )
        
        # Plot confusion matrix
        cm = np.array(evaluation_results['confusion_matrix'])
        inference_engine.plot_confusion_matrix(
            cm, save_path="results/inference_confusion_matrix.png"
        )
        
        # Plot confidence distribution
        inference_engine.plot_confidence_distribution(
            confidences, save_path="results/confidence_distribution.png"
        )
        
        # Save inference results
        inference_engine.save_inference_results(evaluation_results)
        
        return {
            "evaluation_results": evaluation_results,
            "predictions": predicted_classes,
            "confidences": confidences,
            "true_classes": true_classes
        }
    
    return {}

def create_visualizations(config: Config, training_results: Dict[str, Any], inference_results: Dict[str, Any]):
    """
    Create comprehensive visualizations
    
    Args:
        config: Configuration object
        training_results: Training results dictionary
        inference_results: Inference results dictionary
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create visualizer
    visualizer = create_visualizer(config)
    
    # Plot training history
    if 'history' in training_results:
        visualizer.plot_training_history(
            training_results['history'],
            save_path="results/training_history.png"
        )
    
    # Create training report
    if 'test_results' in training_results:
        visualizer.create_training_report(
            training_results['history'],
            training_results['test_results']
        )
    
    # Plot inference results if available
    if 'evaluation_results' in inference_results:
        eval_results = inference_results['evaluation_results']
        
        # Plot confusion matrix
        if 'confusion_matrix' in eval_results:
            cm = np.array(eval_results['confusion_matrix'])
            visualizer.plot_confusion_matrix(
                cm, save_path="results/visualization_confusion_matrix.png"
            )
        
        # Plot class accuracy
        if 'per_class_accuracy' in eval_results:
            visualizer.plot_class_accuracy(
                eval_results['per_class_accuracy'],
                save_path="results/visualization_class_accuracy.png"
            )
    
    print("Visualizations completed!")

def main():
    """Main function to run the complete training pipeline"""
    parser = argparse.ArgumentParser(description="CIFAR-10 Distributed Training")
    parser.add_argument("--model-type", type=str, default="resnet", 
                       choices=["resnet", "cnn", "efficientnet"],
                       help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--load-model", type=str, default=None,
                       help="Path to load existing model")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and only run inference")
    parser.add_argument("--skip-inference", action="store_true",
                       help="Skip inference and only train")
    parser.add_argument("--skip-visualization", action="store_true",
                       help="Skip visualization")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Update configuration based on arguments
    config = Config()
    config.MODEL_TYPE = args.model_type
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    
    # Print configuration
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Model Type: {config.MODEL_TYPE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Distribution Strategy: {config.STRATEGY}")
    print("="*60)
    
    # Training phase
    training_results = {}
    if not args.skip_training:
        training_results = train_model(config, args.load_model)
    
    # Inference phase
    inference_results = {}
    if not args.skip_inference:
        model_path = args.load_model or config.BEST_MODEL_PATH
        if os.path.exists(model_path):
            inference_results = run_inference(config, model_path)
        else:
            print(f"Model not found at {model_path}. Skipping inference.")
    
    # Visualization phase
    if not args.skip_visualization:
        create_visualizations(config, training_results, inference_results)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED")
    print("="*60)
    
    if training_results:
        print(f"Training completed in {training_results.get('training_time', 0):.2f} seconds")
        if 'test_results' in training_results:
            test_acc = training_results['test_results'].get('accuracy', 0)
            print(f"Test Accuracy: {test_acc:.4f}")
    
    if inference_results:
        if 'evaluation_results' in inference_results:
            eval_acc = inference_results['evaluation_results'].get('overall_accuracy', 0)
            print(f"Inference Accuracy: {eval_acc:.4f}")
    
    print(f"Results saved to: results/")
    print(f"Model saved to: {config.BEST_MODEL_PATH}")
    print("="*60)

if __name__ == "__main__":
    main() 