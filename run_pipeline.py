"""
Burnout Prediction Pipeline - Main Execution Script
Run this file to execute the complete training pipeline
"""

import sys
from pathlib import Path

# Add src directory to Python path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.main import main

if __name__ == "__main__":
    print("Starting Burnout Prediction Pipeline...\n")
    output = main()
    print("\nPipeline execution completed successfully!")
