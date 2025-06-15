#!/usr/bin/env python3
"""
Flask Dashboard for OSU AI Replay Maker Training Monitoring

This dashboard provides real-time monitoring of the training process including:
- Current epoch and progress
- Loss graphs over time
- Training metrics
- System status
"""

import os
import json
import time
import psutil
import platform
from datetime import datetime
from flask import Flask, render_template, jsonify
import threading
import re
from collections import deque
try:
    import GPUtil
except ImportError:
    GPUtil = None
    print("GPUtil not available. GPU monitoring will be disabled.")
try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
except ImportError:
    nvml = None
    print("nvidia-ml-py not available. Advanced GPU monitoring will be disabled.")
try:
    import torch
except ImportError:
    torch = None
    print("PyTorch not available. CUDA memory monitoring will be disabled.")

app = Flask(__name__)

# Global variables to store training data
training_data = {
    'current_epoch': 0,
    'total_epochs': 0,
    'current_batch': 0,
    'total_batches': 0,
    'loss_history': deque(maxlen=1000),  # Keep last 1000 loss values
    'epoch_losses': [],
    'learning_rate': 0.0,
    'status': 'Not Started',
    'start_time': None,
    'last_update': None,
    'best_loss': float('inf'),
    'total_samples': 0,
    'validation_loss': None,
    'train_accuracy': None,
    'validation_accuracy': None,
    'samples_per_second': 0,
    'gradient_norm': None,
    'model_parameters': 0,
    'dataset_size': 0
}

# System performance data
system_data = {
    'cpu_percent': deque(maxlen=100),
    'memory_percent': deque(maxlen=100),
    'memory_used_gb': deque(maxlen=100),
    'memory_total_gb': 0,
    'gpu_utilization': deque(maxlen=100),
    'gpu_memory_percent': deque(maxlen=100),
    'gpu_memory_used_gb': deque(maxlen=100),
    'gpu_memory_total_gb': 0,
    'gpu_temperature': deque(maxlen=100),
    'disk_io_read': deque(maxlen=100),
    'disk_io_write': deque(maxlen=100),
    'network_sent': deque(maxlen=100),
    'network_recv': deque(maxlen=100),
    'timestamps': deque(maxlen=100),
    'gpu_count': 0,
    'gpu_names': [],
    'system_info': {}
}

# Log file path - adjust this to match your training script's log output
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training.log')

def get_system_info():
    """Get system information"""
    try:
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0],
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True)
        }
        
        # Get GPU information
        if GPUtil:
            gpus = GPUtil.getGPUs()
            system_data['gpu_count'] = len(gpus)
            system_data['gpu_names'] = [gpu.name for gpu in gpus]
            if gpus:
                system_data['gpu_memory_total_gb'] = gpus[0].memoryTotal / 1024
        
        # Get memory information
        memory = psutil.virtual_memory()
        system_data['memory_total_gb'] = memory.total / (1024**3)
        
        system_data['system_info'] = system_info
        return system_info
    except Exception as e:
        print(f"Error getting system info: {e}")
        return {}

def collect_system_metrics():
    """Collect current system performance metrics"""
    try:
        timestamp = time.time()
        system_data['timestamps'].append(timestamp)
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        system_data['cpu_percent'].append(cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        system_data['memory_percent'].append(memory.percent)
        system_data['memory_used_gb'].append(memory.used / (1024**3))
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        if disk_io:
            system_data['disk_io_read'].append(disk_io.read_bytes / (1024**2))  # MB
            system_data['disk_io_write'].append(disk_io.write_bytes / (1024**2))  # MB
        
        # Network metrics
        network_io = psutil.net_io_counters()
        if network_io:
            system_data['network_sent'].append(network_io.bytes_sent / (1024**2))  # MB
            system_data['network_recv'].append(network_io.bytes_recv / (1024**2))  # MB
        
        # GPU metrics
        if GPUtil:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                system_data['gpu_utilization'].append(gpu.load * 100)
                system_data['gpu_memory_percent'].append(gpu.memoryUtil * 100)
                system_data['gpu_memory_used_gb'].append(gpu.memoryUsed / 1024)
                system_data['gpu_temperature'].append(gpu.temperature)
        
        # PyTorch CUDA memory if available
        if torch and torch.cuda.is_available():
            try:
                torch_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                torch_memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
                # Store in system_data for API access
                system_data['torch_memory_allocated'] = torch_memory_allocated
                system_data['torch_memory_reserved'] = torch_memory_reserved
            except Exception as e:
                pass
                
    except Exception as e:
        print(f"Error collecting system metrics: {e}")

def parse_log_line(line):
    """Parse a log line to extract training information"""
    global training_data
    
    # Parse epoch information - updated to handle log format with timestamps
    epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
    if epoch_match:
        training_data['current_epoch'] = int(epoch_match.group(1))
        training_data['total_epochs'] = int(epoch_match.group(2))
        training_data['status'] = 'Training'
    
    # Parse batch information - updated to handle log format with timestamps
    batch_match = re.search(r'Batch (\d+)/(\d+)', line)
    if batch_match:
        training_data['current_batch'] = int(batch_match.group(1))
        training_data['total_batches'] = int(batch_match.group(2))
    
    # Parse loss information - updated to handle log format with timestamps
    loss_match = re.search(r'Loss: ([0-9.]+)', line)
    if loss_match:
        loss_value = float(loss_match.group(1))
        timestamp = time.time()
        training_data['loss_history'].append({
            'timestamp': timestamp,
            'loss': loss_value,
            'epoch': training_data['current_epoch']
        })
        
        # Update best loss
        if loss_value < training_data['best_loss']:
            training_data['best_loss'] = loss_value
    
    # Parse validation loss - updated to handle log format with timestamps
    val_loss_match = re.search(r'Val Loss: ([0-9.]+)', line)
    if val_loss_match:
        training_data['validation_loss'] = float(val_loss_match.group(1))
    
    # Parse train loss (alternative format)
    train_loss_match = re.search(r'Train Loss: ([0-9.]+)', line)
    if train_loss_match:
        training_data['train_loss'] = float(train_loss_match.group(1))
    
    # Parse accuracy metrics - updated to handle log format with timestamps
    train_acc_match = re.search(r'Train Acc: ([0-9.]+)', line)
    if train_acc_match:
        training_data['train_accuracy'] = float(train_acc_match.group(1))
    
    val_acc_match = re.search(r'Val Acc: ([0-9.]+)', line)
    if val_acc_match:
        training_data['validation_accuracy'] = float(val_acc_match.group(1))
    
    # Parse learning rate - updated to handle scientific notation
    lr_match = re.search(r'LR: ([0-9.e-]+)', line)
    if lr_match:
        training_data['learning_rate'] = float(lr_match.group(1))
    
    # Parse samples per second - updated to handle log format with timestamps
    sps_match = re.search(r'Samples/sec: ([0-9.]+)', line)
    if sps_match:
        training_data['samples_per_second'] = float(sps_match.group(1))
    
    # Parse gradient norm - updated to handle scientific notation
    grad_norm_match = re.search(r'Grad Norm: ([0-9.e+-]+)', line)
    if grad_norm_match:
        training_data['gradient_norm'] = float(grad_norm_match.group(1))
    
    # Parse model parameters - updated to handle log format with timestamps
    params_match = re.search(r'Model Parameters: ([0-9,]+)', line)
    if params_match:
        training_data['model_parameters'] = int(params_match.group(1).replace(',', ''))
    
    # Parse dataset size - updated to handle log format with timestamps
    dataset_match = re.search(r'Dataset Size: ([0-9,]+)', line)
    if dataset_match:
        training_data['dataset_size'] = int(dataset_match.group(1).replace(',', ''))
    
    # Update last update time
    training_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def monitor_log_file():
    """Monitor the log file for new entries"""
    print(f"Starting log monitoring for: {LOG_FILE}")
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} not found. Creating empty file...")
        open(LOG_FILE, 'a').close()
    else:
        print(f"Log file {LOG_FILE} exists. File size: {os.path.getsize(LOG_FILE)} bytes")
    
    # Read existing log file content first to catch up
    lines_processed = 0
    try:
        with open(LOG_FILE, 'r') as f:
            for line in f:
                parse_log_line(line.strip())
                lines_processed += 1
        print(f"Processed {lines_processed} existing log lines")
    except Exception as e:
        print(f"Error reading existing log file: {e}")
    
    # Now monitor for new entries
    print("Starting real-time log monitoring...")
    with open(LOG_FILE, 'r') as f:
        # Go to end of file
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            if line:
                print(f"New log line: {line.strip()[:100]}...")  # Debug print
                parse_log_line(line.strip())
            else:
                time.sleep(0.1)  # Wait for new data

def monitor_system_metrics():
    """Monitor system metrics in a separate thread"""
    while True:
        collect_system_metrics()
        time.sleep(2)  # Collect metrics every 2 seconds

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """API endpoint to get current training status"""
    # Calculate progress percentage
    epoch_progress = 0
    if training_data['total_epochs'] > 0:
        epoch_progress = (training_data['current_epoch'] / training_data['total_epochs']) * 100
    
    batch_progress = 0
    if training_data['total_batches'] > 0:
        batch_progress = (training_data['current_batch'] / training_data['total_batches']) * 100
    
    # Calculate estimated time remaining
    eta = 'Unknown'
    elapsed_time = 'Unknown'
    if training_data['start_time'] and training_data['current_epoch'] > 0:
        elapsed = time.time() - training_data['start_time']
        elapsed_time = f"{int(elapsed // 3600):02d}:{int((elapsed % 3600) // 60):02d}:{int(elapsed % 60):02d}"
        avg_time_per_epoch = elapsed / training_data['current_epoch']
        remaining_epochs = training_data['total_epochs'] - training_data['current_epoch']
        eta_seconds = avg_time_per_epoch * remaining_epochs
        eta = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
    
    return jsonify({
        'current_epoch': training_data['current_epoch'],
        'total_epochs': training_data['total_epochs'],
        'current_batch': training_data['current_batch'],
        'total_batches': training_data['total_batches'],
        'epoch_progress': round(epoch_progress, 1),
        'batch_progress': round(batch_progress, 1),
        'learning_rate': training_data['learning_rate'],
        'status': training_data['status'],
        'last_update': training_data['last_update'],
        'best_loss': training_data['best_loss'] if training_data['best_loss'] != float('inf') else None,
        'validation_loss': training_data['validation_loss'],
        'train_accuracy': training_data['train_accuracy'],
        'validation_accuracy': training_data['validation_accuracy'],
        'samples_per_second': training_data['samples_per_second'],
        'gradient_norm': training_data['gradient_norm'],
        'model_parameters': training_data['model_parameters'],
        'dataset_size': training_data['dataset_size'],
        'eta': eta,
        'elapsed_time': elapsed_time
    })

@app.route('/api/system')
def get_system_metrics():
    """API endpoint to get current system performance metrics"""
    # Get current metrics
    current_metrics = {}
    
    if system_data['cpu_percent']:
        current_metrics['cpu_percent'] = system_data['cpu_percent'][-1]
    if system_data['memory_percent']:
        current_metrics['memory_percent'] = system_data['memory_percent'][-1]
        current_metrics['memory_used_gb'] = system_data['memory_used_gb'][-1]
    
    current_metrics['memory_total_gb'] = system_data['memory_total_gb']
    
    # GPU metrics
    if system_data['gpu_utilization']:
        current_metrics['gpu_utilization'] = system_data['gpu_utilization'][-1]
        current_metrics['gpu_memory_percent'] = system_data['gpu_memory_percent'][-1]
        current_metrics['gpu_memory_used_gb'] = system_data['gpu_memory_used_gb'][-1]
        current_metrics['gpu_temperature'] = system_data['gpu_temperature'][-1]
    
    current_metrics['gpu_memory_total_gb'] = system_data['gpu_memory_total_gb']
    current_metrics['gpu_count'] = system_data['gpu_count']
    current_metrics['gpu_names'] = system_data['gpu_names']
    
    # PyTorch CUDA memory
    if 'torch_memory_allocated' in system_data:
        current_metrics['torch_memory_allocated'] = system_data['torch_memory_allocated']
        current_metrics['torch_memory_reserved'] = system_data['torch_memory_reserved']
    
    # Disk and Network I/O
    if system_data['disk_io_read']:
        current_metrics['disk_io_read_mb'] = system_data['disk_io_read'][-1]
        current_metrics['disk_io_write_mb'] = system_data['disk_io_write'][-1]
    
    if system_data['network_sent']:
        current_metrics['network_sent_mb'] = system_data['network_sent'][-1]
        current_metrics['network_recv_mb'] = system_data['network_recv'][-1]
    
    # System info
    current_metrics['system_info'] = system_data['system_info']
    
    return jsonify(current_metrics)

@app.route('/api/system_history')
def get_system_history():
    """API endpoint to get system performance history for charts"""
    # Convert timestamps to readable format
    timestamps = [datetime.fromtimestamp(ts).strftime('%H:%M:%S') for ts in list(system_data['timestamps'])]
    
    return jsonify({
        'timestamps': timestamps,
        'cpu_percent': list(system_data['cpu_percent']),
        'memory_percent': list(system_data['memory_percent']),
        'memory_used_gb': list(system_data['memory_used_gb']),
        'gpu_utilization': list(system_data['gpu_utilization']),
        'gpu_memory_percent': list(system_data['gpu_memory_percent']),
        'gpu_memory_used_gb': list(system_data['gpu_memory_used_gb']),
        'gpu_temperature': list(system_data['gpu_temperature']),
        'disk_io_read': list(system_data['disk_io_read']),
        'disk_io_write': list(system_data['disk_io_write']),
        'network_sent': list(system_data['network_sent']),
        'network_recv': list(system_data['network_recv'])
    })

@app.route('/api/loss_data')
def get_loss_data():
    """API endpoint to get loss history for plotting"""
    # Convert deque to list for JSON serialization
    loss_data = list(training_data['loss_history'])
    
    # Limit to last 500 points for performance
    if len(loss_data) > 500:
        loss_data = loss_data[-500:]
    
    return jsonify({
        'loss_history': loss_data,
        'epoch_losses': training_data['epoch_losses']
    })

def start_monitoring():
    """Start the log and system monitoring in separate threads"""
    training_data['start_time'] = time.time()
    
    # Initialize system info
    get_system_info()
    
    # Start log monitoring thread
    log_monitor_thread = threading.Thread(target=monitor_log_file, daemon=True)
    log_monitor_thread.start()
    
    # Start system metrics monitoring thread
    system_monitor_thread = threading.Thread(target=monitor_system_metrics, daemon=True)
    system_monitor_thread.start()

if __name__ == '__main__':
    print("Starting OSU AI Training Dashboard...")
    print(f"Monitoring log file: {LOG_FILE}")
    print("Dashboard will be available at: http://localhost:5000")
    
    # Start log monitoring
    start_monitoring()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)