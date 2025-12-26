
import sys

def test_imports():
    print("\n" + "="*70)
    print("TESTING PACKAGE IMPORTS")
    print("="*70)
    
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('sklearn', 'scikit-learn'),
        ('skimage', 'scikit-image'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('joblib', 'joblib'),
    ]
    
    optional_packages = [
        ('xgboost', 'XGBoost'),
        ('torch', 'PyTorch'),
    ]
    
    failed = []
    
    print("\nRequired Packages:")
    print("-" * 70)
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"✓ {package_name:<20} - OK")
        except ImportError as e:
            print(f"✗ {package_name:<20} - FAILED")
            failed.append(package_name)
    
    print("\nOptional Packages:")
    print("-" * 70)
    for module_name, package_name in optional_packages:
        try:
            __import__(module_name)
            print(f"✓ {package_name:<20} - OK")
        except ImportError:
            print(f"○ {package_name:<20} - Not installed (optional)")
    
    return failed

def test_project_files():
    print("\n" + "="*70)
    print("TESTING PROJECT FILES")
    print("="*70)
    
    import os
    
    required_files = [
        'configs.py',
        'preprocessing.py',
        'features.py',
        'feature_selection.py',
        'utils.py',
        'prepare_dataset.py',
        'create_crops.py',
        'train.py',
        'evaluate.py',
        'realtime_detection.py',
        'main.py',
        'requirements.txt',
        'README.md',
    ]
    
    missing = []
    
    print("\nProject Files:")
    print("-" * 70)
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✓ {filename:<30} - Found")
        else:
            print(f"✗ {filename:<30} - Missing")
            missing.append(filename)
    
    return missing

def test_python_version():
    print("\n" + "="*70)
    print("TESTING PYTHON VERSION")
    print("="*70)
    
    version = sys.version_info
    print(f"\nPython Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 7:
        print("✓ Python version is compatible (>= 3.7)")
        return True
    else:
        print("✗ Python version is too old (requires >= 3.7)")
        return False

def test_opencv():
    print("\n" + "="*70)
    print("TESTING OPENCV")
    print("="*70)
    
    try:
        import cv2
        import numpy as np
        
        print(f"\nOpenCV Version: {cv2.__version__}")
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print("✓ OpenCV is working correctly")
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("LEGO BRICK FINDER - INSTALLATION TEST")
    print("="*70)
    
    python_ok = test_python_version()
    
    failed_imports = test_imports()
    
    opencv_ok = test_opencv()
    
    missing_files = test_project_files()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    if python_ok and len(failed_imports) == 0 and opencv_ok and len(missing_files) == 0:
        print("\n✓ All tests passed! Your installation is complete.")
        print("\nYou can now run the project:")
        print("  python main.py")
        return True
    else:
        print("\n✗ Some tests failed. Please fix the issues:")
        
        if not python_ok:
            print("\n  - Upgrade Python to version 3.7 or higher")
        
        if failed_imports:
            print("\n  - Install missing packages:")
            print("    pip install -r requirements.txt")
        
        if not opencv_ok:
            print("\n  - Reinstall OpenCV:")
            print("    pip install opencv-python --upgrade")
        
        if missing_files:
            print(f"\n  - Missing files: {', '.join(missing_files)}")
        
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)