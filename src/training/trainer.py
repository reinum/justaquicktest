"""Main trainer class for osu! replay generation model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import wandb
from tqdm import tqdm
import json
import pickle

from ..config.model_config import TrainingConfig, ModelConfig
from ..models.transformer import OsuTransformer
from .loss import ReplayLoss
from .metrics import CombinedMetrics
from .scheduler import get_scheduler


class OsuTrainer:
    """Main trainer class for osu! replay generation."""
    
    def __init__(self, 
                 model: OsuTransformer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: TrainingConfig,
                 model_config: ModelConfig,
                 device: torch.device,
                 logger: Optional[logging.Logger] = None):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_config = model_config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_and_metrics()
        self._setup_mixed_precision()
        self._setup_checkpointing()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        self.training_history = defaultdict(list)
        
        # Curriculum learning
        self.current_difficulty = 0.0
        self.difficulty_schedule = self._create_difficulty_schedule()
        
        # Early stopping
        self.patience_counter = 0
        
        self.logger.info(f"Trainer initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    def _setup_optimizer(self):
        """Setup optimizer with parameter groups."""
        # Separate parameters for different learning rates
        embedding_params = []
        transformer_params = []
        output_params = []
        
        for name, param in self.model.named_parameters():
            if 'embedding' in name or 'pos_encoding' in name:
                embedding_params.append(param)
            elif 'output' in name or 'head' in name:
                output_params.append(param)
            else:
                transformer_params.append(param)
        
        param_groups = [
            {'params': embedding_params, 'lr': self.config.learning_rate * 0.5, 'name': 'embeddings'},
            {'params': transformer_params, 'lr': self.config.learning_rate, 'name': 'transformer'},
            {'params': output_params, 'lr': self.config.learning_rate * 2.0, 'name': 'outputs'}
        ]
        
        if self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                param_groups,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_config = {
            'type': 'onecycle',
            'max_lr': self.config.learning_rate,
            'total_steps': len(self.train_loader) * self.config.max_epochs,
            'pct_start': 0.1,
            'anneal_strategy': 'cos',
            'cycle_momentum': True
        }
        
        self.scheduler = get_scheduler(self.optimizer, scheduler_config)
        self.use_scheduler = True
    
    def _setup_loss_and_metrics(self):
        """Setup loss function and metrics."""
        self.criterion = ReplayLoss(
            cursor_weight=1.0,
            key_weight=0.5,
            temporal_weight=0.3,
            cursor_loss_type='mse',
            key_loss_type='bce',
            smoothness_weight=0.1
        )
        
        self.metrics = CombinedMetrics()
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
    
    def _setup_checkpointing(self):
        """Setup checkpointing directories."""
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_model_path = self.checkpoint_dir / 'best_model.pt'
        self.latest_model_path = self.checkpoint_dir / 'latest_model.pt'
    
    def _create_difficulty_schedule(self) -> List[float]:
        """Create curriculum learning difficulty schedule."""
        if not self.config.use_curriculum_learning:
            return [1.0] * self.config.max_epochs
        
        # Gradually increase difficulty from 0.1 to 1.0
        schedule = []
        for epoch in range(self.config.max_epochs):
            progress = epoch / max(1, self.config.max_epochs - 1)
            difficulty = 0.1 + 0.9 * min(1.0, progress * 1.5)  # Slightly faster ramp-up
            schedule.append(difficulty)
        
        return schedule
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)
        
        # Track performance metrics
        epoch_start_time = time.time()
        total_samples = 0
        
        # Update curriculum difficulty
        if self.current_epoch < len(self.difficulty_schedule):
            self.current_difficulty = self.difficulty_schedule[self.current_epoch]
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Count samples for performance tracking
            batch_size = batch['cursor_data'].size(0)
            total_samples += batch_size
            
            # Apply curriculum learning (filter by difficulty)
            if self.config.use_curriculum_learning:
                batch = self._apply_curriculum_filter(batch, self.current_difficulty)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        cursor_data=batch['cursor_data'],
                        beatmap_data=batch['beatmap_data'],
                        timing_data=batch['timing_data'],
                        key_data=batch['key_data'],
                        accuracy_target=batch['accuracy_target'],
                        slider_data=batch.get('slider_data', None)
                    )
                    loss_dict = self.criterion(outputs, batch)
                    total_loss = loss_dict['total_loss']
            else:
                outputs = self.model(
                    cursor_data=batch['cursor_data'],
                    beatmap_data=batch['beatmap_data'],
                    timing_data=batch['timing_data'],
                    key_data=batch['key_data'],
                    accuracy_target=batch['accuracy_target'],
                    slider_data=batch.get('slider_data', None)
                )
                loss_dict = self.criterion(outputs, batch)
                total_loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                
                # Calculate gradient norm before clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    self._gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 self.config.gradient_clip_norm).item()
                else:
                    # Calculate gradient norm without clipping
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    self._gradient_norm = total_norm ** (1. / 2)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                
                # Calculate gradient norm before clipping
                if self.config.gradient_clip_norm > 0:
                    self._gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 self.config.gradient_clip_norm).item()
                else:
                    # Calculate gradient norm without clipping
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    self._gradient_norm = total_norm ** (1. / 2)
                
                self.optimizer.step()
            
            # Update scheduler
            if self.use_scheduler:
                self.scheduler.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()
            
            # Calculate metrics
            with torch.no_grad():
                self.metrics.update(outputs, batch)
                batch_metrics = self.metrics.compute()
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{current_lr:.2e}",
                'diff': f"{self.current_difficulty:.2f}"
            })
            
            # Log batch progress for dashboard (every 10 batches to avoid spam)
            if batch_idx % 10 == 0:
                self.logger.info(f"Batch {batch_idx + 1}/{len(self.train_loader)} - Loss: {total_loss.item():.6f} - LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            self.global_step += 1
            
            # Log to wandb
            if self.config.use_wandb and self.global_step % self.config.log_interval == 0:
                wandb.log({
                    'train/loss': total_loss.item(),
                    'train/learning_rate': current_lr,
                    'train/difficulty': self.current_difficulty,
                    'train/epoch': self.current_epoch,
                    'train/step': self.global_step
                })
        
        # Calculate performance metrics
        epoch_time = time.time() - epoch_start_time
        self._samples_per_second = total_samples / epoch_time if epoch_time > 0 else 0
        
        # Average losses and metrics
        num_batches = len(self.train_loader)
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = defaultdict(float)
        val_metrics = defaultdict(float)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            cursor_data=batch['cursor_data'],
                            beatmap_data=batch['beatmap_data'],
                            timing_data=batch['timing_data'],
                            key_data=batch['key_data'],
                            accuracy_target=batch['accuracy_target'],
                            slider_data=batch.get('slider_data', None)
                        )
                        loss_dict = self.criterion(outputs, batch)
                else:
                    outputs = self.model(
                        cursor_data=batch['cursor_data'],
                        beatmap_data=batch['beatmap_data'],
                        timing_data=batch['timing_data'],
                        key_data=batch['key_data'],
                        accuracy_target=batch['accuracy_target'],
                        slider_data=batch.get('slider_data', None)
                    )
                    loss_dict = self.criterion(outputs, batch)
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    val_losses[key] += value.item()
                
                # Calculate metrics
                self.metrics.update(outputs, batch)
                batch_metrics = self.metrics.compute()
                for key, value in batch_metrics.items():
                    val_metrics[key] += value
        
        # Average losses and metrics
        num_batches = len(self.val_loader)
        avg_losses = {k: v / num_batches for k, v in val_losses.items()}
        avg_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    def _apply_curriculum_filter(self, batch: Dict[str, torch.Tensor], 
                                difficulty: float) -> Dict[str, torch.Tensor]:
        """Apply curriculum learning by filtering batch based on difficulty."""
        # For now, just return the batch as-is
        # In a full implementation, you might filter by beatmap difficulty,
        # sequence length, or other complexity measures
        return batch
    
    def save_checkpoint(self, filepath: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.use_scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics,
            'training_history': dict(self.training_history),
            'config': self.config.__dict__,
            'model_config': self.model_config.__dict__
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            torch.save(checkpoint, self.best_model_path)
        
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: Path) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.use_scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_metrics = checkpoint['best_metrics']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_results = self.train_epoch()
            
            # Validate
            val_results = self.validate()
            
            # Log results in dashboard-compatible format
            self.logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs} - Train Loss: {train_results['total_loss']:.6f} - Val Loss: {val_results['total_loss']:.6f} - LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Log additional metrics for dashboard
            if 'accuracy' in train_results:
                self.logger.info(f"Train Acc: {train_results['accuracy']:.4f}")
            if 'accuracy' in val_results:
                self.logger.info(f"Val Acc: {val_results['accuracy']:.4f}")
            
            # Log performance metrics
            if hasattr(self, '_samples_per_second'):
                self.logger.info(f"Samples/sec: {self._samples_per_second:.2f}")
            
            if hasattr(self, '_gradient_norm'):
                self.logger.info(f"Grad Norm: {self._gradient_norm:.4e}")
            
            # Log model info (once per training)
            if epoch == 0:
                total_params = sum(p.numel() for p in self.model.parameters())
                self.logger.info(f"Model Parameters: {total_params:,}")
                self.logger.info(f"Dataset Size: {len(self.train_loader.dataset):,}")
            
            # Also log individual components for debugging
            self.logger.info(f"  Train Loss: {train_results['total_loss']:.4f}")
            self.logger.info(f"  Val Loss: {val_results['total_loss']:.4f}")
            
            # Update training history
            for key, value in train_results.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_results.items():
                self.training_history[f'val_{key}'].append(value)
            
            # Check for best model
            current_val_loss = val_results['total_loss']
            is_best = current_val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = current_val_loss
                self.best_metrics = val_results.copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoints
            if (epoch + 1) % self.config.save_interval == 0:
                # Save with epoch number in filename
                epoch_checkpoint_path = self.checkpoint_dir / f'model_epoch_{epoch + 1:04d}.pt'
                self.save_checkpoint(epoch_checkpoint_path, is_best=is_best)
                # Also save as latest for compatibility
                self.save_checkpoint(self.latest_model_path, is_best=False)
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    **{f'train/{k}': v for k, v in train_results.items()},
                    **{f'val/{k}': v for k, v in val_results.items()},
                    'epoch': epoch + 1
                })
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {self.config.early_stopping_patience} epochs without improvement")
                break
        
        # Save final checkpoint
        self.save_checkpoint(self.checkpoint_dir / 'final_model.pt')
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return dict(self.training_history)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'device': str(self.device)
        }