import torch
import sys

def check_gpu():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU Count: {device_count}")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        
        # Simple tensor operation test
        try:
            x = torch.tensor([1.0, 2.0]).cuda()
            print("Tensor operation on GPU successful.")
        except Exception as e:
            print(f"Tensor operation failed: {e}")
    else:
        print("WARNING: CUDA is not available. Please install PyTorch with CUDA support.")

if __name__ == "__main__":
    check_gpu()
