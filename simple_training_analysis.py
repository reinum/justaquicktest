import torch
import numpy as np

def analyze_training():
    checkpoint_path = "model_epoch_0033.pt"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"=== Training Analysis for {checkpoint_path} ===")
        
        # Basic info
        epoch = checkpoint.get('epoch', 'Unknown')
        global_step = checkpoint.get('global_step', 'Unknown')
        best_val_loss = checkpoint.get('best_val_loss', 'Unknown')
        
        print(f"Current epoch: {epoch}")
        print(f"Global step: {global_step}")
        print(f"Best validation loss: {best_val_loss}")
        
        # Training history analysis
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            print(f"\n=== Training History ===")
            
            if isinstance(history, dict):
                # Check for loss trends
                train_losses = history.get('train_loss', [])
                val_losses = history.get('val_loss', [])
                
                print(f"Training epochs recorded: {len(train_losses)}")
                
                if len(train_losses) > 0:
                    print(f"First train loss: {train_losses[0]:.6f}")
                    print(f"Latest train loss: {train_losses[-1]:.6f}")
                    
                    if len(train_losses) > 1:
                        improvement = train_losses[0] - train_losses[-1]
                        improvement_pct = (improvement / train_losses[0]) * 100
                        print(f"Training loss improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
                        
                        # Check recent trend (last 5 epochs)
                        if len(train_losses) >= 5:
                            recent_losses = train_losses[-5:]
                            recent_trend = recent_losses[-1] - recent_losses[0]
                            print(f"Recent trend (last 5 epochs): {recent_trend:.6f}")
                            if recent_trend < 0:
                                print("✓ Loss is still decreasing")
                            else:
                                print("⚠ Loss may be plateauing or increasing")
                
                if len(val_losses) > 0:
                    print(f"\nValidation losses recorded: {len(val_losses)}")
                    print(f"First val loss: {val_losses[0]:.6f}")
                    print(f"Latest val loss: {val_losses[-1]:.6f}")
                    
                    if len(val_losses) > 1:
                        val_improvement = val_losses[0] - val_losses[-1]
                        val_improvement_pct = (val_improvement / val_losses[0]) * 100
                        print(f"Validation loss improvement: {val_improvement:.6f} ({val_improvement_pct:.2f}%)")
                
                # Check other metrics
                for metric_name in ['val_accuracy', 'val_timing_accuracy', 'val_rhythm_correlation']:
                    if metric_name in history:
                        metric_values = history[metric_name]
                        if len(metric_values) > 0:
                            print(f"\n{metric_name}: {metric_values[0]:.4f} → {metric_values[-1]:.4f}")
        
        # Model health assessment
        print(f"\n=== Training Health Assessment ===")
        
        health_score = 0
        max_score = 5
        
        # Check 1: Model has reasonable parameters
        model_state = checkpoint.get('model_state_dict', {})
        if model_state:
            health_score += 1
            print("✓ Model state available")
        
        # Check 2: Training progressed
        if isinstance(epoch, int) and epoch > 0:
            health_score += 1
            print(f"✓ Training progressed to epoch {epoch}")
        
        # Check 3: Has optimizer state
        if 'optimizer_state_dict' in checkpoint:
            health_score += 1
            print("✓ Optimizer state preserved")
        
        # Check 4: Loss improvement
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            if isinstance(history, dict) and 'train_loss' in history:
                train_losses = history['train_loss']
                if len(train_losses) > 1 and train_losses[-1] < train_losses[0]:
                    health_score += 1
                    print("✓ Training loss improved")
        
        # Check 5: Reasonable validation loss
        if isinstance(best_val_loss, (int, float)) and not np.isnan(best_val_loss) and not np.isinf(best_val_loss):
            health_score += 1
            print(f"✓ Valid best validation loss: {best_val_loss:.6f}")
        
        print(f"\nOverall Health Score: {health_score}/{max_score}")
        
        if health_score >= 4:
            print("🟢 Training appears healthy - continue training or use for generation")
        elif health_score >= 2:
            print("🟡 Training shows some issues - monitor closely")
        else:
            print("🔴 Training appears problematic - consider restarting")
            
    except Exception as e:
        print(f"Error analyzing checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_training()