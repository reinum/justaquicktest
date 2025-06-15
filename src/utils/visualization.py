"""Visualization utilities for the osu! AI replay maker."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from matplotlib.animation import FuncAnimation

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Progress"
) -> None:
    """Plot training curves for loss and metrics.
    
    Args:
        train_losses: List of training losses
        val_losses: Optional list of validation losses
        metrics: Optional dictionary of metric name -> values
        save_path: Optional path to save the plot
        title: Plot title
    """
    # Determine number of subplots
    num_plots = 1 + (1 if metrics else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]
    
    # Plot losses
    ax_loss = axes[0]
    epochs = range(1, len(train_losses) + 1)
    
    ax_loss.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        ax_loss.plot(val_epochs, val_losses, label='Validation Loss', linewidth=2)
    
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training and Validation Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # Plot metrics if provided
    if metrics and num_plots > 1:
        ax_metrics = axes[1]
        
        for metric_name, values in metrics.items():
            metric_epochs = range(1, len(values) + 1)
            ax_metrics.plot(metric_epochs, values, label=metric_name, linewidth=2)
        
        ax_metrics.set_xlabel('Epoch')
        ax_metrics.set_ylabel('Metric Value')
        ax_metrics.set_title('Training Metrics')
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def plot_replay_comparison(
    original_replay: List[Dict[str, Any]],
    generated_replay: List[Dict[str, Any]],
    beatmap_objects: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[str] = None,
    title: str = "Replay Comparison"
) -> None:
    """Compare original and generated replays visually.
    
    Args:
        original_replay: Original replay data
        generated_replay: Generated replay data
        beatmap_objects: Optional beatmap hit objects
        save_path: Optional path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract coordinates
    orig_x = [event.get('x', 0) for event in original_replay]
    orig_y = [event.get('y', 0) for event in original_replay]
    orig_time = [event.get('time', i) for i, event in enumerate(original_replay)]
    
    gen_x = [event.get('x', 0) for event in generated_replay]
    gen_y = [event.get('y', 0) for event in generated_replay]
    gen_time = [event.get('time', i) for i, event in enumerate(generated_replay)]
    
    # Plot 1: Cursor paths
    ax1 = axes[0, 0]
    ax1.plot(orig_x, orig_y, label='Original', alpha=0.7, linewidth=1)
    ax1.plot(gen_x, gen_y, label='Generated', alpha=0.7, linewidth=1)
    
    # Add hit objects if provided
    if beatmap_objects:
        hit_x = [obj.get('x', 0) for obj in beatmap_objects]
        hit_y = [obj.get('y', 0) for obj in beatmap_objects]
        ax1.scatter(hit_x, hit_y, c='red', s=30, alpha=0.6, label='Hit Objects')
    
    ax1.set_xlim(0, 512)
    ax1.set_ylim(0, 384)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Cursor Paths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: X coordinate over time
    ax2 = axes[0, 1]
    ax2.plot(orig_time, orig_x, label='Original', alpha=0.7)
    ax2.plot(gen_time, gen_x, label='Generated', alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X Position')
    ax2.set_title('X Position Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Y coordinate over time
    ax3 = axes[1, 0]
    ax3.plot(orig_time, orig_y, label='Original', alpha=0.7)
    ax3.plot(gen_time, gen_y, label='Generated', alpha=0.7)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Y Position')
    ax3.set_title('Y Position Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Speed comparison
    ax4 = axes[1, 1]
    
    # Calculate speeds
    orig_speeds = calculate_speeds(orig_x, orig_y, orig_time)
    gen_speeds = calculate_speeds(gen_x, gen_y, gen_time)
    
    ax4.plot(orig_time[1:], orig_speeds, label='Original', alpha=0.7)
    ax4.plot(gen_time[1:], gen_speeds, label='Generated', alpha=0.7)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Speed (pixels/ms)')
    ax4.set_title('Cursor Speed Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Replay comparison saved to: {save_path}")
    
    plt.show()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    input_tokens: Optional[List[str]] = None,
    output_tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Attention Heatmap"
) -> None:
    """Plot attention weights as a heatmap.
    
    Args:
        attention_weights: 2D array of attention weights
        input_tokens: Optional list of input token labels
        output_tokens: Optional list of output token labels
        save_path: Optional path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=input_tokens if input_tokens else False,
        yticklabels=output_tokens if output_tokens else False,
        cmap='Blues',
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.xlabel('Input Tokens')
    plt.ylabel('Output Tokens')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention heatmap saved to: {save_path}")
    
    plt.show()


def plot_loss_landscape(
    loss_values: np.ndarray,
    param1_range: np.ndarray,
    param2_range: np.ndarray,
    param1_name: str = "Parameter 1",
    param2_name: str = "Parameter 2",
    save_path: Optional[str] = None
) -> None:
    """Plot loss landscape for parameter exploration.
    
    Args:
        loss_values: 2D array of loss values
        param1_range: Range of first parameter
        param2_range: Range of second parameter
        param1_name: Name of first parameter
        param2_name: Name of second parameter
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D heatmap
    im1 = ax1.contourf(param1_range, param2_range, loss_values, levels=20, cmap='viridis')
    ax1.set_xlabel(param1_name)
    ax1.set_ylabel(param2_name)
    ax1.set_title('Loss Landscape (2D)')
    plt.colorbar(im1, ax=ax1, label='Loss')
    
    # 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(param1_range, param2_range)
    surf = ax2.plot_surface(X, Y, loss_values, cmap='viridis', alpha=0.8)
    ax2.set_xlabel(param1_name)
    ax2.set_ylabel(param2_name)
    ax2.set_zlabel('Loss')
    ax2.set_title('Loss Landscape (3D)')
    plt.colorbar(surf, ax=ax2, label='Loss', shrink=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss landscape saved to: {save_path}")
    
    plt.show()


def plot_model_architecture(
    layer_sizes: List[int],
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """Visualize model architecture.
    
    Args:
        layer_sizes: List of layer sizes
        layer_names: Optional list of layer names
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate positions
    max_size = max(layer_sizes)
    layer_positions = np.linspace(0, 10, len(layer_sizes))
    
    for i, (pos, size) in enumerate(zip(layer_positions, layer_sizes)):
        # Draw layer as rectangle
        height = (size / max_size) * 6  # Scale height
        rect = patches.Rectangle(
            (pos - 0.3, 3 - height/2), 0.6, height,
            linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add size label
        ax.text(pos, 3 - height/2 - 0.5, f'{size}', ha='center', va='top', fontsize=10)
        
        # Add layer name if provided
        if layer_names and i < len(layer_names):
            ax.text(pos, 3 + height/2 + 0.3, layer_names[i], ha='center', va='bottom', fontsize=10, rotation=45)
        
        # Draw connections to next layer
        if i < len(layer_sizes) - 1:
            next_pos = layer_positions[i + 1]
            ax.arrow(pos + 0.3, 3, next_pos - pos - 0.6, 0, 
                    head_width=0.1, head_length=0.1, fc='gray', ec='gray')
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Model Architecture', fontsize=16, pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model architecture saved to: {save_path}")
    
    plt.show()


def create_replay_animation(
    replay_data: List[Dict[str, Any]],
    beatmap_objects: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[str] = None,
    fps: int = 60,
    duration_ms: int = 10000
) -> None:
    """Create an animated visualization of a replay.
    
    Args:
        replay_data: Replay event data
        beatmap_objects: Optional beatmap hit objects
        save_path: Optional path to save the animation
        fps: Frames per second
        duration_ms: Duration in milliseconds
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Setup plot
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 384)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Replay Animation')
    ax.grid(True, alpha=0.3)
    
    # Add hit objects if provided
    if beatmap_objects:
        for obj in beatmap_objects:
            circle = patches.Circle(
                (obj.get('x', 0), obj.get('y', 0)), 
                radius=20, 
                facecolor='red', 
                alpha=0.3
            )
            ax.add_patch(circle)
    
    # Initialize cursor
    cursor, = ax.plot([], [], 'bo', markersize=8, label='Cursor')
    trail, = ax.plot([], [], 'b-', alpha=0.5, linewidth=2, label='Trail')
    
    # Animation data
    x_data, y_data = [], []
    
    def animate(frame):
        if frame < len(replay_data):
            event = replay_data[frame]
            x, y = event.get('x', 0), event.get('y', 0)
            
            x_data.append(x)
            y_data.append(y)
            
            # Update cursor position
            cursor.set_data([x], [y])
            
            # Update trail (last 100 points)
            trail_length = min(100, len(x_data))
            trail.set_data(x_data[-trail_length:], y_data[-trail_length:])
        
        return cursor, trail
    
    # Create animation
    frames = min(len(replay_data), duration_ms // (1000 // fps))
    anim = FuncAnimation(fig, animate, frames=frames, interval=1000//fps, blit=True)
    
    ax.legend()
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Replay animation saved to: {save_path}")
    
    plt.show()


def plot_evaluation_metrics(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Evaluation Metrics"
) -> None:
    """Plot evaluation metrics as a bar chart.
    
    Args:
        metrics: Dictionary of metric names and values
        save_path: Optional path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color='skyblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if needed
    if len(metric_names) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation metrics saved to: {save_path}")
    
    plt.show()


def calculate_speeds(x_coords: List[float], y_coords: List[float], times: List[float]) -> List[float]:
    """Calculate cursor speeds between consecutive points.
    
    Args:
        x_coords: X coordinates
        y_coords: Y coordinates
        times: Time values
        
    Returns:
        List of speeds in pixels per millisecond
    """
    speeds = []
    
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        dt = times[i] - times[i-1]
        
        if dt > 0:
            distance = np.sqrt(dx**2 + dy**2)
            speed = distance / dt
            speeds.append(speed)
        else:
            speeds.append(0.0)
    
    return speeds


def save_plots_to_pdf(plots: List[plt.Figure], output_path: str) -> None:
    """Save multiple plots to a single PDF file.
    
    Args:
        plots: List of matplotlib figures
        output_path: Path to output PDF file
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(output_path) as pdf:
        for fig in plots:
            pdf.savefig(fig, bbox_inches='tight')
    
    print(f"Plots saved to PDF: {output_path}")