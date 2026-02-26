import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch

def check_gpu():
    print("=" * 60)
    print("GPU Configuration Check")
    print("=" * 60)
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # GPU details
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Current device
        current_device = torch.cuda.current_device()
        print(f"\nCurrent Device: {current_device}")
        print(f"Current Device Name: {torch.cuda.get_device_name(current_device)}")
        
        # Memory info
        print(f"\nMemory Allocated: {torch.cuda.memory_allocated(current_device) / 1024**3:.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(current_device) / 1024**3:.2f} GB")
        
        # Test tensor creation
        print("\nTesting GPU tensor creation...")
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"✓ Successfully created tensor on GPU")
        print(f"  Tensor device: {test_tensor.device}")
        
        # Recommended settings for RTX 3050
        print("\n" + "=" * 60)
        print("Recommended Settings for RTX 3050 (4GB VRAM):")
        print("=" * 60)
        print("Batch Size: 64 (can try 128 if memory allows)")
        print("Num Workers: 4")
        print("Mixed Precision: Enabled (fp16)")
        print("Gradient Accumulation: 1-2 steps if needed")
        
    else:
        print("\n⚠ WARNING: CUDA not available!")
        print("Please check:")
        print("1. NVIDIA drivers installed")
        print("2. PyTorch installed with CUDA support")
        print("3. GPU is enabled in system")
    
    print("=" * 60)

if __name__ == "__main__":
    check_gpu()
