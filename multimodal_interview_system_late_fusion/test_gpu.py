"""
GPU Availability and Configuration Test
"""
import torch
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

def test_gpu():
    """Test GPU availability and configuration"""
    
    print("=" * 70)
    print("GPU CONFIGURATION TEST")
    print("=" * 70)
    
    # Check CUDA availability
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA is not available!")
        print("   Possible reasons:")
        print("   - PyTorch not installed with CUDA support")
        print("   - NVIDIA GPU drivers not installed")
        print("   - No NVIDIA GPU in system")
        print("\n   Install PyTorch with CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    # CUDA details
    print(f"2. CUDA Version: {torch.version.cuda}")
    print(f"3. cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"4. Number of GPUs: {torch.cuda.device_count()}")
    
    # GPU details
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"   - Name: {torch.cuda.get_device_name(i)}")
        print(f"   - Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   - Total Memory: {total_memory:.2f} GB")
        
        # Current memory usage
        if torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   - Allocated Memory: {allocated:.2f} GB")
            print(f"   - Reserved Memory: {reserved:.2f} GB")
    
    # Test tensor operations
    print("\n5. Testing GPU Operations:")
    try:
        device = torch.device('cuda:0')
        
        # Create test tensor
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        z = torch.matmul(x, y)
        
        print("   ✓ Matrix multiplication successful")
        
        # Test mixed precision
        with torch.amp.autocast('cuda'):
            z_amp = torch.matmul(x, y)
        
        print("   ✓ Mixed precision (AMP) successful")
        
        # Clean up
        del x, y, z, z_amp
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ GPU operation failed: {e}")
        return False
    
    # PyTorch configuration
    print("\n6. PyTorch Configuration:")
    print(f"   - PyTorch Version: {torch.__version__}")
    print(f"   - cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"   - cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    
    # Recommendations
    print("\n7. Recommendations for Training:")
    print("   ✓ Use mixed precision training (AMP) for faster training")
    print("   ✓ Set pin_memory=True in DataLoader")
    print("   ✓ Use gradient accumulation for larger effective batch size")
    
    if torch.cuda.device_count() > 1:
        print("   ✓ Multiple GPUs detected - consider DataParallel or DistributedDataParallel")
    
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_mem < 8:
        print(f"   ⚠ GPU memory is {total_mem:.1f}GB - consider reducing batch size")
    elif total_mem >= 12:
        print(f"   ✓ GPU memory is {total_mem:.1f}GB - sufficient for training")
    
    print("\n" + "=" * 70)
    print("✓ GPU is ready for training!")
    print("=" * 70)
    
    return True


def test_model_on_gpu():
    """Test loading model on GPU"""
    print("\n" + "=" * 70)
    print("MODEL GPU TEST")
    print("=" * 70)
    
    try:
        from models.multimodal_model import MultimodalInterviewModel
        from utils.helpers import load_config
        
        print("\n1. Loading model configuration...")
        config = load_config('config/model_config.yaml')
        
        print("2. Initializing model...")
        model = MultimodalInterviewModel(config)
        
        print("3. Moving model to GPU...")
        device = torch.device('cuda:0')
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Model size: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")
        
        # Test forward pass
        print("\n4. Testing forward pass on GPU...")
        batch_size = 2
        
        # Create dummy inputs
        video = torch.randn(batch_size, 30, 3, 224, 224, device=device)
        audio = torch.randn(batch_size, 300, 120, device=device)
        text = {
            'input_ids': torch.randint(0, 1000, (batch_size, 128), device=device),
            'attention_mask': torch.ones(batch_size, 128, device=device)
        }
        
        # Forward pass
        with torch.no_grad():
            outputs = model(video, audio, text)
        
        print("   ✓ Forward pass successful")
        print(f"   - Output keys: {list(outputs.keys())}")
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n5. GPU Memory Usage:")
        print(f"   - Allocated: {allocated:.2f} GB")
        print(f"   - Reserved: {reserved:.2f} GB")
        
        # Clean up
        del model, video, audio, text, outputs
        torch.cuda.empty_cache()
        
        print("\n✓ Model is ready for GPU training!")
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    print("\nTesting GPU configuration for Multimodal Interview System (Late Fusion)\n")
    
    # Test GPU
    gpu_ok = test_gpu()
    
    if gpu_ok:
        # Test model
        print("\n")
        model_ok = test_model_on_gpu()
        
        if model_ok:
            print("\n" + "=" * 70)
            print("✓ ALL TESTS PASSED - SYSTEM IS READY FOR GPU TRAINING!")
            print("=" * 70)
            print("\nYou can now run:")
            print("  python main_train.py --config config/training_config.yaml \\")
            print("                       --model_config config/model_config.yaml \\")
            print("                       --data_dir path/to/mdpe_dataset \\")
            print("                       --seumld_data_dir D:\\Grad\\Datasets\\SEUMLD\\SEUMLD \\")
            print("                       --use_combined")
            sys.exit(0)
        else:
            print("\n❌ Model test failed")
            sys.exit(1)
    else:
        print("\n❌ GPU test failed")
        sys.exit(1)

