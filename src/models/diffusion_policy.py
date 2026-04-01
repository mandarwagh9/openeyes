"""Diffusion Policy for robot manipulation.

Implements diffusion-based action prediction for robot control.
Based on "Diffusion Models for Robotic Manipulation" (Columbia/TRI).

Requirements:
    - torch
    - torch.nn
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time


@dataclass
class DiffusionAction:
    """Action predicted by diffusion policy."""
    action: np.ndarray
    confidence: float
    denoising_steps: int


class DiffusionModel(nn.Module):
    """Simple diffusion model for action prediction.
    
    Implements a minimal diffusion model that predicts robot actions
    from visual observations.
    """
    
    def __init__(
        self,
        obs_dim: int = 512,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._num_steps = num_steps
        
        betas = torch.linspace(beta_start, beta_end, num_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        
        self._encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self._denoiser = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        self._noise_proj = nn.Linear(action_dim, hidden_dim)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Training forward pass."""
        B = actions.shape[0]
        
        t = torch.randint(0, self._num_steps, (B,), device=actions.device)
        
        noise = torch.randn_like(actions)
        alpha_t = self.alphas_cumprod[t].view(B, 1)
        noisy_actions = torch.sqrt(alpha_t) * actions + torch.sqrt(1 - alpha_t) * noise
        
        obs_enc = self._encoder(obs)
        noise_feat = self._noise_proj(noise)
        
        combined = torch.cat([noisy_actions, obs_enc], dim=-1)
        predicted_noise = self._denoiser(combined)
        
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def denoise(
        self,
        obs: torch.Tensor,
        num_steps: Optional[int] = None,
        action_dim: int = 7,
    ) -> torch.Tensor:
        """Generate actions via denoising.
        
        Args:
            obs: Observation tensor
            num_steps: Number of denoising steps
            action_dim: Dimension of action space
            
        Returns:
            Denoised action tensor
        """
        num_steps = num_steps or self._num_steps
        
        B = obs.shape[0] if obs.dim() > 1 else 1
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        actions = torch.randn(B, action_dim, device=obs.device)
        
        obs_enc = self._encoder(obs)
        
        for t in reversed(range(num_steps)):
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            if t > 0:
                alpha_cumprod_prev = self.alphas_cumprod[t - 1]
            else:
                alpha_cumprod_prev = torch.ones_like(alpha_cumprod_t)
            
            noise = torch.randn_like(actions)
            
            noise_feat = self._noise_proj(noise)
            combined = torch.cat([actions, obs_enc], dim=-1)
            predicted_noise = self._denoiser(combined)
            
            coef1 = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
            coef2 = torch.sqrt(alpha_cumprod_prev) * beta_t / (1 - alpha_cumprod_t)
            
            actions = actions + coef1.view(B, 1) * predicted_noise + coef2.view(B, 1) * noise
            
            if t > 0:
                actions = actions + torch.sqrt(beta_t) * noise
        
        return actions


class DiffusionPolicy:
    """Diffusion Policy for robot manipulation.
    
    Provides high-level interface for diffusion-based action prediction.
    
    Features:
    - Multi-step denoising for action generation
    - Observation history for temporal context
    - Action smoothing and filtering
    """
    
    def __init__(
        self,
        obs_dim: int = 512,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 10,
        device: str = "cuda",
        action_horizon: int = 8,
        observation_horizon: int = 2,
        smoothing_window: int = 3,
    ):
        self._device = device
        self._action_dim = action_dim
        self._num_inference_steps = num_inference_steps
        self._action_horizon = action_horizon
        self._observation_horizon = observation_horizon
        
        self._model = DiffusionModel(
            obs_dim=obs_dim,
            action_dim=action_dim * action_horizon,
            hidden_dim=hidden_dim,
            num_steps=num_diffusion_steps,
        ).to(device)
        
        self._model.eval()
        
        self._observation_history: deque = deque(maxlen=observation_horizon)
        self._action_history: deque = deque(maxlen=smoothing_window)
        
        self._is_loaded = False
        
    def load_weights(self, path: str) -> bool:
        """Load model weights from checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            True if successful
        """
        try:
            state_dict = torch.load(path, map_location=self._device)
            self._model.load_state_dict(state_dict)
            self._is_loaded = True
            return True
        except Exception:
            return False
    
    def save_weights(self, path: str) -> bool:
        """Save model weights to checkpoint.
        
        Args:
            path: Path to save checkpoint
            
        Returns:
            True if successful
        """
        try:
            torch.save(self._model.state_dict(), path)
            return True
        except Exception:
            return False
    
    def add_observation(self, observation: np.ndarray) -> None:
        """Add observation to history.
        
        Args:
            observation: Observation vector or features
        """
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        
        if observation.shape[-1] != self._model._obs_dim:
            observation = F.adaptive_avg_pool1d(
                observation.T, self._model._obs_dim
            ).T
        
        self._observation_history.append(observation.to(self._device))
    
    def predict_action(
        self,
        observation: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """Predict action from observation.
        
        Args:
            observation: Optional observation (uses history if not provided)
            context: Optional context (task description, goals, etc.)
            
        Returns:
            Predicted action vector
        """
        if observation is not None:
            self.add_observation(observation)
        
        if len(self._observation_history) == 0:
            return None
        
        obs_tensor = torch.cat(list(self._observation_history), dim=0)
        
        if obs_tensor.shape[0] > 1:
            obs_tensor = obs_tensor.mean(dim=0, keepdim=True)
        
        with torch.no_grad():
            flat_actions = self._model.denoise(
                obs_tensor,
                num_steps=self._num_inference_steps,
                action_dim=self._action_dim * self._action_horizon,
            )
        
        actions = flat_actions.view(-1, self._action_horizon, self._action_dim)
        
        action = actions[0, 0].cpu().numpy()
        
        if len(self._action_history) > 0:
            smoothed = np.mean(
                np.array(self._action_history[-self._action_horizon:]),
                axis=0
            )
            action = 0.7 * action + 0.3 * smoothed
        
        self._action_history.append(action)
        
        return action
    
    def reset_history(self) -> None:
        """Clear observation and action history."""
        self._observation_history.clear()
        self._action_history.clear()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model weights are loaded."""
        return self._is_loaded
    
    @property
    def action_dim(self) -> int:
        """Get action dimension."""
        return self._action_dim
    
    @property
    def action_horizon(self) -> int:
        """Get action horizon."""
        return self._action_horizon
    
    def train(self, mode: bool = True) -> None:
        """Set model training mode."""
        self._model.train(mode)


class DiffusionPolicyWrapper:
    """High-level wrapper for diffusion policy with VLA integration.
    
    Combines diffusion policy with VLA models for enhanced manipulation.
    """
    
    def __init__(
        self,
        vla_model: Any = None,
        diffusion_policy: Optional[DiffusionPolicy] = None,
        device: str = "cuda",
        use_fusion: bool = True,
    ):
        self._vla_model = vla_model
        self._diffusion = diffusion_policy
        self._device = device
        self._use_fusion = use_fusion
        
        self._logger = get_logger(__name__)
        
    def set_vla_model(self, model: Any) -> None:
        """Set VLA model."""
        self._vla_model = model
    
    def set_diffusion_policy(self, policy: DiffusionPolicy) -> None:
        """Set diffusion policy."""
        self._diffusion = policy
    
    def predict(
        self,
        observation: np.ndarray,
        instruction: str = "",
    ) -> Optional[np.ndarray]:
        """Predict action using VLA + Diffusion fusion.
        
        Args:
            observation: Visual observation
            instruction: Natural language instruction
            
        Returns:
            Predicted action
        """
        vla_action = None
        
        if self._vla_model is not None and hasattr(self._vla_model, "predict_action"):
            try:
                vla_result = self._vla_model.predict_action(
                    observation,
                    instruction
                )
                if vla_result is not None:
                    vla_action = vla_result.values
            except Exception:
                pass
        
        diffusion_action = None
        if self._diffusion is not None:
            try:
                diffusion_action = self._diffusion.predict_action(observation)
            except Exception:
                pass
        
        if vla_action is None and diffusion_action is None:
            return None
        
        if self._use_fusion and vla_action is not None and diffusion_action is not None:
            return 0.5 * vla_action + 0.5 * diffusion_action
        elif vla_action is not None:
            return vla_action
        else:
            return diffusion_action
    
    def update_observation(self, observation: np.ndarray) -> None:
        """Update observation history."""
        if self._diffusion is not None:
            self._diffusion.add_observation(observation)
    
    def reset(self) -> None:
        """Reset all policies."""
        if self._diffusion is not None:
            self._diffusion.reset_history()


def create_diffusion_policy(
    obs_dim: int = 512,
    action_dim: int = 7,
    num_inference_steps: int = 10,
    device: str = "cuda",
) -> DiffusionPolicy:
    """Factory function to create diffusion policy.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        num_inference_steps: Denoising steps for inference
        device: Device for model
        
    Returns:
        DiffusionPolicy instance
    """
    return DiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_inference_steps=num_inference_steps,
        device=device,
    )


from src.utils.logger import get_logger