"""
Test script to verify installation and basic functionality
"""

import os
import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    required_packages = [
        'tensorflow',
        'tensorflow_datasets',
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'pandas',
        'tqdm',
        'PIL'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("\nAll packages imported successfully!")
        return True

def test_config():
    """Test configuration module"""
    print("\nTesting configuration module...")
    
    try:
        from config import Config
        
        # Test configuration creation
        config = Config()
        print("✓ Configuration created successfully")
        
        # Test configuration methods
        model_config = config.get_model_config()
        training_config = config.get_training_config()
        distributed_config = config.get_distributed_config()
        callback_config = config.get_callback_config()
        
        print("✓ Configuration methods working")
        
        # Test directory creation
        config.create_directories()
        print("✓ Directories created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_data_loader():
    """Test data loader module"""
    print("\nTesting data loader module...")
    
    try:
        from config import Config
        from data_loader import create_data_loader
        
        config = Config()
        data_loader = create_data_loader(config)
        
        print("✓ Data loader created successfully")
        
        # Test dataset info
        info = data_loader.get_dataset_info()
        print(f"✓ Dataset info retrieved: {info['dataset_name']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def test_models():
    """Test models module"""
    print("\nTesting models module...")
    
    try:
        from config import Config
        from models import create_model
        
        config = Config()
        
        # Test different model types
        model_types = ["resnet", "cnn", "efficientnet"]
        
        for model_type in model_types:
            config.MODEL_TYPE = model_type
            model = create_model(config)
            print(f"✓ {model_type.upper()} model created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Models test failed: {e}")
        return False

def test_callbacks():
    """Test callbacks module"""
    print("\nTesting callbacks module...")
    
    try:
        from config import Config
        from callbacks import create_callbacks
        
        config = Config()
        callbacks = create_callbacks(config)
        
        print(f"✓ {len(callbacks)} callbacks created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Callbacks test failed: {e}")
        return False

def test_visualization():
    """Test visualization module"""
    print("\nTesting visualization module...")
    
    try:
        from config import Config
        from visualization import create_visualizer
        
        config = Config()
        visualizer = create_visualizer(config)
        
        print("✓ Visualizer created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def test_tensorflow_setup():
    """Test TensorFlow setup and GPU availability"""
    print("\nTesting TensorFlow setup...")
    
    try:
        import tensorflow as tf
        
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
        else:
            print("⚠ No GPUs detected - will use CPU")
        
        # Test basic TensorFlow operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = a + b
        print(f"✓ Basic TensorFlow operations working: {c.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("CIFAR-10 DISTRIBUTED TRAINING - INSTALLATION TEST")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("TensorFlow Setup", test_tensorflow_setup),
        ("Configuration", test_config),
        ("Data Loader", test_data_loader),
        ("Models", test_models),
        ("Callbacks", test_callbacks),
        ("Visualization", test_visualization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Installation is complete.")
        print("\nYou can now run the training script:")
        print("python main.py --model-type resnet --epochs 10")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check TensorFlow installation")
        print("3. Verify GPU drivers (if using GPU)")
    
    print("="*60)

if __name__ == "__main__":
    main() 