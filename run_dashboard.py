#!/usr/bin/env python3
"""
Script to run the training dashboard

Usage:
    python run_dashboard.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    print("🌐 Starting OSU AI Training Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("📝 Monitoring log file: training.log")
    print("⏹️  Press Ctrl+C to stop")
    print()
    
    try:
        # Run the dashboard
        subprocess.run([sys.executable, "dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except FileNotFoundError:
        print("❌ dashboard.py not found. Make sure you're in the correct directory.")
        return 1
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())