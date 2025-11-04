"""
GPU Availability Checker for Warehouse Robot RL Project
Run this before training to verify GPU is properly configured
"""

import torch
import sys

def check_gpu():
    """Check GPU availability and configuration"""

    print("=" * 70)
    print("GPU CONFIGURATION CHECK")
    print("=" * 70)

    # PyTorch version
    print(f"\n[PyTorch Info]")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA Compiled: {torch.version.cuda if torch.cuda.is_available() else 'No'}")

    # CUDA availability
    print(f"\n[CUDA Status]")
    cuda_available = torch.cuda.is_available()
    print(f"  Available: {'YES' if cuda_available else 'NO'}")

    if cuda_available:
        # GPU details
        print(f"\n[GPU Details]")
        print(f"  Device Count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}:")
            print(f"    Name: {torch.cuda.get_device_name(i)}")

            props = torch.cuda.get_device_properties(i)
            print(f"    Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")

            # Memory status
            print(f"    Current Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"    Current Reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")

        # Tensor Core support (for mixed precision)
        capability = torch.cuda.get_device_capability(0)
        has_tensor_cores = capability[0] >= 7
        print(f"\n[Performance Features]")
        print(f"  Tensor Cores (Mixed Precision): {'YES (Ampere/Turing+)' if has_tensor_cores else 'NO'}")
        print(f"  cuDNN Available: {torch.backends.cudnn.is_available()}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")

        # Quick GPU test
        print(f"\n[GPU Test]")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            print(f"  Matrix multiplication test: PASSED")
            print(f"  GPU is working correctly!")
        except Exception as e:
            print(f"  Matrix multiplication test: FAILED")
            print(f"  Error: {e}")
            return False

        # Recommendations
        print(f"\n[Training Recommendations]")
        if has_tensor_cores:
            print(f"  - Mixed precision training: RECOMMENDED (will speed up training)")
            print(f"  - Expected speedup: 1.5-2x faster than FP32")
        else:
            print(f"  - Mixed precision training: NOT RECOMMENDED (older GPU architecture)")

        print(f"  - Batch size: Start with 64, increase if memory allows")
        print(f"  - Expected training time: ~2-4 hours for full training")

    else:
        print(f"\n[CPU Mode]")
        print(f"  Training will use CPU (much slower than GPU)")
        print(f"  Expected training time: ~10-20 hours for full training")
        print(f"\n[To Enable GPU]")
        print(f"  1. Install PyTorch with CUDA:")
        print(f"     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print(f"  2. Make sure NVIDIA drivers are installed")
        print(f"  3. Restart this script")

    print("\n" + "=" * 70)

    return cuda_available

if __name__ == "__main__":
    gpu_available = check_gpu()

    if gpu_available:
        print("\n[STATUS] GPU is ready! You can start training with: python main.py")
        sys.exit(0)
    else:
        print("\n[STATUS] No GPU detected. Training will be slow on CPU.")
        print("         Consider installing PyTorch with CUDA support.")
        sys.exit(1)
