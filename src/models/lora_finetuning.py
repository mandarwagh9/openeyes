"""LoRA fine-tuning support for VLA models.

Provides parameter-efficient fine-tuning for VLA customization
without requiring full model retraining.

Requirements:
    - peft (Parameter-Efficient Fine-Tuning)
    - transformers
    - torch
"""

from typing import Optional, List, Dict, Any, Callable
import numpy as np
from dataclasses import dataclass
import torch
from pathlib import Path


try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from src.utils.logger import get_logger


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "SEQ_CLS"


class LoRAFineTuner:
    """LoRA fine-tuner for VLA models.
    
    Enables efficient fine-tuning of VLA models using Low-Rank Adaptation.
    This allows customization without modifying the original model weights.
    
    Benefits:
    - 10-100x fewer trainable parameters
    - Faster training (2-4x)
    - Lower memory requirements
    - Easy to swap back to base model
    """
    
    def __init__(
        self,
        model: Any,
        config: Optional[LoRAConfig] = None,
        device: str = "cuda",
    ):
        self._logger = get_logger(__name__)
        self._model = model
        self._config = config or LoRAConfig()
        self._device = device
        
        self._peft_model = None
        self._is_prepared = False
        self._is_available = PEFT_AVAILABLE
        
        if not self._is_available:
            self._logger.warning(
                "PEFT not available. Install with: pip install peft"
            )
    
    def prepare_for_training(
        self,
        trainable_layers: Optional[List[str]] = None,
    ) -> Any:
        """Prepare model for LoRA training.
        
        Args:
            trainable_layers: List of layer names to apply LoRA to
            
        Returns:
            PEFT-wrapped model
        """
        if not self._is_available:
            self._logger.error("Cannot prepare: PEFT not installed")
            return self._model
        
        if self._is_prepared:
            self._logger.warning("Model already prepared for LoRA")
            return self._peft_model
        
        try:
            target_modules = self._config.target_modules or self._get_default_targets()
            
            lora_config = LoraConfig(
                r=self._config.rank,
                lora_alpha=self._config.alpha,
                lora_dropout=self._config.dropout,
                target_modules=target_modules,
                bias=self._config.bias,
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
            )
            
            self._peft_model = get_peft_model(self._model, lora_config)
            self._peft_model.print_trainable_parameters()
            
            self._is_prepared = True
            self._logger.info("Model prepared for LoRA training")
            
            return self._peft_model
            
        except Exception as e:
            self._logger.error(f"Failed to prepare LoRA: {e}")
            return self._model
    
    def _get_default_targets(self) -> List[str]:
        """Get default target modules for LoRA."""
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    
    def get_peft_model(self) -> Optional[Any]:
        """Get the PEFT-wrapped model."""
        if not self._is_prepared:
            self._logger.warning("Model not prepared. Call prepare_for_training() first.")
        return self._peft_model
    
    def save_adapter(self, path: str) -> bool:
        """Save LoRA adapter weights.
        
        Args:
            path: Directory to save adapter
            
        Returns:
            True if successful
        """
        if not self._is_prepared or self._peft_model is None:
            self._logger.error("Model not prepared for training")
            return False
        
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            self._peft_model.save_pretrained(str(save_path))
            self._logger.info(f"Adapter saved to {path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save adapter: {e}")
            return False
    
    def load_adapter(self, path: str) -> bool:
        """Load LoRA adapter weights.
        
        Args:
            path: Directory containing adapter
            
        Returns:
            True if successful
        """
        if not self._is_available:
            self._logger.error("PEFT not available")
            return False
        
        try:
            from peft import PeftModel
            
            self._peft_model = PeftModel.from_pretrained(
                self._model,
                path,
            )
            self._is_prepared = True
            
            self._logger.info(f"Adapter loaded from {path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load adapter: {e}")
            return False
    
    def merge_adapter(self) -> Any:
        """Merge LoRA weights into base model.
        
        Returns:
            Merged model
        """
        if not self._is_prepared or self._peft_model is None:
            self._logger.error("No adapter to merge")
            return self._model
        
        try:
            merged = self._peft_model.merge_weights()
            self._logger.info("Adapter merged into model")
            return merged
            
        except Exception as e:
            self._logger.error(f"Failed to merge adapter: {e}")
            return self._model
    
    def disable_adapter(self) -> None:
        """Disable LoRA adapter (use base model)."""
        if self._peft_model is not None:
            self._peft_model.disable_adapter()
            self._logger.info("Adapter disabled")
    
    def enable_adapter(self) -> None:
        """Enable LoRA adapter."""
        if self._peft_model is not None:
            self._peft_model.enable_adapter()
            self._logger.info("Adapter enabled")
    
    @property
    def is_prepared(self) -> bool:
        """Check if model is prepared for training."""
        return self._is_prepared
    
    @property
    def is_available(self) -> bool:
        """Check if PEFT is available."""
        return self._is_available


class VLALoRATrainer:
    """Trainer for VLA models with LoRA.
    
    Provides training loop and utilities for fine-tuning VLA models.
    """
    
    def __init__(
        self,
        model: Any,
        config: Optional[LoRAConfig] = None,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
    ):
        self._logger = get_logger(__name__)
        
        self._fine_tuner = LoRAFineTuner(model, config, device)
        self._device = device
        self._learning_rate = learning_rate
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        
        self._optimizer = None
        self._scheduler = None
        
    def prepare_model(self) -> Any:
        """Prepare model for training."""
        return self._fine_tuner.prepare_for_training()
    
    def setup_optimizer(self) -> None:
        """Setup optimizer and scheduler."""
        if not self._fine_tuner.is_prepared:
            self._logger.error("Model not prepared")
            return
        
        peft_model = self._fine_tuner.get_peft_model()
        if peft_model is None:
            return
        
        self._optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, peft_model.parameters()),
            lr=self._learning_rate,
        )
        
        self._scheduler = torch.optim.lr_scheduler.LinearLR(
            self._optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=self._num_epochs,
        )
        
        self._logger.info("Optimizer and scheduler configured")
    
    def train_step(
        self,
        batch: Dict[str, Any],
    ) -> float:
        """Single training step.
        
        Args:
            batch: Training batch with 'images', 'instructions', 'actions'
            
        Returns:
            Loss value
        """
        if self._optimizer is None:
            self.setup_optimizer()
        
        peft_model = self._fine_tuner.get_peft_model()
        if peft_model is None:
            return 0.0
        
        try:
            self._optimizer.zero_grad()
            
            images = batch.get("images", [])
            instructions = batch.get("instructions", [])
            actions = batch.get("actions", torch.zeros(len(images), 7))
            
            outputs = peft_model(
                pixel_values=images,
                input_ids=instructions,
                labels=actions,
            )
            
            loss = outputs.loss
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            
            return loss.item()
            
        except Exception as e:
            self._logger.warning(f"Training step error: {e}")
            return 0.0
    
    def train(
        self,
        train_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_data: Training dataset
            validation_data: Optional validation dataset
            callback: Optional callback called after each epoch
            
        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
        }
        
        if not self._fine_tuner.is_prepared:
            self.prepare_model()
        
        self.setup_optimizer()
        
        for epoch in range(self._num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_data), self._batch_size):
                batch_data = train_data[i:i + self._batch_size]
                
                batch = {
                    "images": torch.stack([
                        torch.from_numpy(d["image"]) 
                        for d in batch_data
                    ]).to(self._device),
                    "instructions": [
                        d.get("instruction", "") 
                        for d in batch_data
                    ],
                    "actions": torch.tensor(
                        [d.get("action", [0]*7) for d in batch_data],
                        device=self._device
                    ),
                }
                
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_loss)
            
            if callback:
                callback(epoch, avg_loss)
            
            self._logger.info(f"Epoch {epoch+1}/{self._num_epochs}: Loss = {avg_loss:.4f}")
        
        return history
    
    def save(self, path: str) -> bool:
        """Save trained adapter."""
        return self._fine_tuner.save_adapter(path)
    
    def load(self, path: str) -> bool:
        """Load trained adapter."""
        return self._fine_tuner.load_adapter(path)


def create_lora_fine_tuner(
    model: Any,
    rank: int = 16,
    alpha: int = 32,
    device: str = "cuda",
) -> LoRAFineTuner:
    """Factory function to create LoRA fine-tuner.
    
    Args:
        model: Base model to fine-tune
        rank: LoRA rank (default 16)
        alpha: LoRA alpha scaling (default 32)
        device: Device for training
        
    Returns:
        LoRAFineTuner instance
    """
    config = LoRAConfig(rank=rank, alpha=alpha)
    return LoRAFineTuner(model, config, device)