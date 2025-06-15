#!/usr/bin/env python3
"""
Unified script to start both the training dashboard and training process.

This script launches:
1. The training dashboard (Flask web server) in a separate process
2. The training script with all its arguments

Usage:
    python train_dashboard.py                    # Use default config
    python train_dashboard.py --config custom.yaml  # Use custom config
    python train_dashboard.py --resume checkpoint.pt  # Resume from checkpoint
    python train_dashboard.py --dataset my_dataset   # Use custom dataset folder
    python train_dashboard.py -d reduced_dataset_100 # Use reduced dataset
    
Dashboard will be available at: http://localhost:5000
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
import signal
import threading


class ProcessManager:
    """Manages the dashboard and training processes."""
    
    def __init__(self):
        self.dashboard_process = None
        self.training_process = None
        self.shutdown_event = threading.Event()
    
    def start_dashboard(self):
        """Start the dashboard process."""
        try:
            print("🌐 Starting training dashboard...")
            self.dashboard_process = subprocess.Popen(
                [sys.executable, "dashboard.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give dashboard time to start
            time.sleep(3)
            
            if self.dashboard_process.poll() is None:
                print("✅ Dashboard started successfully at http://localhost:5000")
                return True
            else:
                stdout, stderr = self.dashboard_process.communicate()
                print(f"❌ Dashboard failed to start:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting dashboard: {e}")
            return False
    
    def start_training(self, args):
        """Start the training process with given arguments."""
        try:
            print("🚀 Starting training process...")
            
            # Build command arguments
            cmd = [sys.executable, "train.py"]
            
            if args.config != 'config/default.yaml':
                cmd.extend(['--config', args.config])
            if args.resume:
                cmd.extend(['--resume', args.resume])
            if args.debug:
                cmd.append('--debug')
            if args.dataset != 'dataset':
                cmd.extend(['--dataset', args.dataset])
            
            print(f"📝 Running command: {' '.join(cmd)}")
            
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream training output in real-time
            def stream_output():
                for line in iter(self.training_process.stdout.readline, ''):
                    if line:
                        print(f"[TRAINING] {line.rstrip()}")
                    if self.shutdown_event.is_set():
                        break
            
            output_thread = threading.Thread(target=stream_output, daemon=True)
            output_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting training: {e}")
            return False
    
    def wait_for_training(self):
        """Wait for training to complete."""
        if self.training_process:
            try:
                return_code = self.training_process.wait()
                if return_code == 0:
                    print("\n✅ Training completed successfully!")
                else:
                    print(f"\n⚠️  Training exited with code: {return_code}")
                return return_code
            except KeyboardInterrupt:
                print("\n⏹️  Training interrupted by user")
                self.shutdown()
                return 1
        return 0
    
    def shutdown(self):
        """Gracefully shutdown both processes."""
        print("\n🛑 Shutting down processes...")
        self.shutdown_event.set()
        
        # Terminate training process
        if self.training_process and self.training_process.poll() is None:
            print("   Stopping training process...")
            try:
                self.training_process.terminate()
                self.training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("   Force killing training process...")
                self.training_process.kill()
            except Exception as e:
                print(f"   Error stopping training: {e}")
        
        # Terminate dashboard process
        if self.dashboard_process and self.dashboard_process.poll() is None:
            print("   Stopping dashboard process...")
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("   Force killing dashboard process...")
                self.dashboard_process.kill()
            except Exception as e:
                print(f"   Error stopping dashboard: {e}")
        
        print("✅ Shutdown complete")


def signal_handler(signum, frame, manager):
    """Handle interrupt signals."""
    manager.shutdown()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Start both training dashboard and training process',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_dashboard.py                           # Default training
  python train_dashboard.py --config my_config.yaml   # Custom config
  python train_dashboard.py --resume checkpoint.pt    # Resume training
  python train_dashboard.py -d my_dataset             # Custom dataset
  python train_dashboard.py --debug                   # Debug mode

The dashboard will be available at: http://localhost:5000
        """
    )
    
    parser.add_argument('--config', '-c', default='config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', '-r', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--dataset', '-d', type=str, default='dataset',
                       help='Path to dataset folder (default: dataset)')
    parser.add_argument('--dashboard-only', action='store_true',
                       help='Start only the dashboard (no training)')
    
    args = parser.parse_args()
    
    print("🎮 osu! AI Replay Maker - Training Dashboard")
    print("=" * 50)
    print()
    
    # Check if required files exist
    if not Path("dashboard.py").exists():
        print("❌ dashboard.py not found in current directory")
        return 1
    
    if not args.dashboard_only and not Path("train.py").exists():
        print("❌ train.py not found in current directory")
        return 1
    
    # Create process manager
    manager = ProcessManager()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, manager))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, manager))
    
    try:
        # Start dashboard
        if not manager.start_dashboard():
            print("❌ Failed to start dashboard")
            return 1
        
        if args.dashboard_only:
            print("\n📊 Dashboard-only mode. Press Ctrl+C to stop.")
            print("🌐 Dashboard available at: http://localhost:5000")
            try:
                # Keep the script running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            # Start training
            if not manager.start_training(args):
                print("❌ Failed to start training")
                manager.shutdown()
                return 1
            
            print("\n🎯 Both processes started successfully!")
            print("🌐 Dashboard: http://localhost:5000")
            print("🚀 Training: Running in background")
            print("\n💡 Press Ctrl+C to stop both processes")
            print("=" * 50)
            print()
            
            # Wait for training to complete
            return_code = manager.wait_for_training()
            
            # Keep dashboard running for a bit after training completes
            if return_code == 0:
                print("\n📊 Training completed! Dashboard will remain available for 30 seconds...")
                print("🌐 View final results at: http://localhost:5000")
                print("💡 Press Ctrl+C to stop dashboard immediately")
                try:
                    time.sleep(30)
                except KeyboardInterrupt:
                    pass
            
            manager.shutdown()
            return return_code
            
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        manager.shutdown()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        manager.shutdown()
        return 1


if __name__ == '__main__':
    sys.exit(main())