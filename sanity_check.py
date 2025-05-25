import os
import importlib
import pandas as pd
import torch

REQUIRED_FILES = [
    'data/dataset_train_p_en.csv',
    'data/dataset_val_p_en.csv',
    'data/dataset_test_p_en.csv'
]

REQUIRED_MODULES = [
    'transformers',
    'sklearn',
    'torch',
    'torchmetrics',
    'pandas',
    'numpy'
]

def check_files_exist():
    print("\n📁 Checking required dataset files:")
    for f in REQUIRED_FILES:
        if not os.path.exists(f):
            print(f"❌ Missing: {f}")
        else:
            print(f"✅ Found: {f}")

def check_python_packages():
    print("\n📦 Checking Python module dependencies:")
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} NOT INSTALLED")

def check_pandas_version():
    print("\n🧪 Checking pandas compatibility:")
    version = pd.__version__
    print(f"ℹ️ pandas version: {version}")
    if version.startswith("2"):
        print("⚠️ pandas 2.x detected: .append() is deprecated. Make sure it's replaced.")
    else:
        print("✅ .append() is still safe to use.")

def check_gpu():
    print("\n🖥️ Checking GPU availability:")
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available. Training will run on CPU.")

def run_preflight():
    print("=== 🚀 Preflight Check for train.py ===")
    check_files_exist()
    check_python_packages()
    check_pandas_version()
    check_gpu()
    print("=== ✅ Preflight Done ===\n")

if __name__ == "__main__":
    run_preflight()
