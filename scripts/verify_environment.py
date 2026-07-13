"""System diagnostics utility for verification of Python, PyTorch, CUDA, and library dependencies.
"""

from __future__ import annotations

import sys
import platform
from pathlib import Path

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)

def main() -> int:
    print_section("DRDO Threat Assessment - System Diagnostics")

    # 1. OS & Python Check
    print(f"[*] Operating System: {platform.system()} ({platform.release()})")
    print(f"[*] Python Version:   {platform.python_version()} (Target: >= 3.9)")
    
    python_ok = sys.version_info >= (3, 9)
    if python_ok:
        print("[+] Python version check: PASSED")
    else:
        print("[-] Python version check: FAILED (Required >= 3.9)")

    # 2. CUDA & PyTorch Check
    print_section("PyTorch & CUDA Diagnostics")
    try:
        import torch
        print(f"[+] PyTorch installed: Version {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"[*] CUDA Available:    {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"[+] GPU Count:         {device_count}")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_properties = torch.cuda.get_device_properties(i)
                total_memory_gb = device_properties.total_memory / (1024 ** 3)
                print(f"    - GPU {i}: {device_name} ({total_memory_gb:.2f} GB VRAM)")
                
            current_device = torch.cuda.current_device()
            print(f"[*] Selected GPU ID:   {current_device}")
        else:
            print("[-] CUDA is not available. PyTorch will execute on CPU (warning: VLM training is extremely slow).")
            
    except ImportError:
        print("[-] PyTorch: NOT INSTALLED")
        cuda_available = False

    # 3. Package Check
    print_section("Required Packages Verification")
    required_packages = [
        "transformers", "peft", "trl", "bitsandbytes", 
        "accelerate", "datasets", "cv2", "PIL", "gradio"
    ]
    
    all_ok = True
    for pkg in required_packages:
        try:
            if pkg == "cv2":
                __import__("cv2")
            elif pkg == "PIL":
                __import__("PIL.Image")
            else:
                __import__(pkg)
            print(f"[+] {pkg:<15} : Available")
        except ImportError:
            print(f"[-] {pkg:<15} : MISSING")
            all_ok = False

    # 4. Final Verdict
    print_section("Diagnostics Summary")
    if python_ok and all_ok:
        if cuda_available:
            print("[SUCCESS] Environment is fully optimized for GPU SFT training & evaluation.")
        else:
            print("[WARNING] Environment is running in CPU-only mode. Offline inference works, but SFT training is blocked.")
        return 0
    else:
        print("[ERROR] Environment setup is incomplete. Please install missing packages.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
