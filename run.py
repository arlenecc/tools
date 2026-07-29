#!/usr/bin/env python3
"""
OpenAI Debug Tool - A GUI application for debugging OpenAI-compatible API services

Features:
- Configure base URL, API key, and model name
- Fetch available models from the service with one click
- Interactive chat interface with streaming responses
- Real-time token generation speed display
- Debug logging panel for troubleshooting
- Persistent configuration storage

Usage:
    python run.py
"""

import sys
import os

# Ensure src directory is in path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

from main import main

if __name__ == "__main__":
    main()
