#!/usr/bin/env python3
"""
Setup script for gprMax Documentation Chatbot
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"Python version: {sys.version}")

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("Error installing dependencies")
        sys.exit(1)

def check_pdf_file():
    """Check if the PDF file exists"""
    pdf_file = "docs-gprmax-com-en-latest.pdf"
    if not os.path.exists(pdf_file):
        print(f"Warning: {pdf_file} not found in the current directory")
        print("The chatbot will use dummy data for testing")
    else:
        print(f"PDF file found: {pdf_file}")

def main():
    """Main setup function"""
    print("Setting up gprMax Documentation Chatbot...")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Check PDF file
    check_pdf_file()
    
    print("\nSetup completed successfully!")
    print("\nTo run the chatbot, use the following command:")
    print("streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 