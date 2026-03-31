from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
from dataclasses import dataclass
import time

from src.camera.types import Detection, BoundingBox
from src.utils.logger import get_logger


try:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class VLAAction:
    """VLA action output."""
    action_type: str
    values: np.ndarray
    confidence: float
    reasoning: str


class SmolVLAWrapper:
    """SmolVLA wrapper for OpenEyes.
    
    SmolVLA is a lightweight VLA model (~450M params) from HuggingFace LeRobot.
    It's designed to run on consumer hardware and Jetson devices.
    
    Model: lerobot/smolvla_base
    
    Input:
        - Camera images (1-2 views)
        - Natural language instruction
        - Optional: robot state (joint positions)
    
    Output:
        - 7-DoF action: [x, y, z, roll, pitch, yaw, gripper]
    
    Requirements:
        pip install torch transformers pillow
    """
    
    MODEL_NAME = "lerobot/smolvla_base"
    
    def __init__(
        self,
        device: str = "cuda",
        model_name: Optional[str] = None,
        quantize: bool = True,
    ):
        self._logger = get_logger(__name__)
        self._device = device
        self._model_name = model_name or self.MODEL_NAME
        self._quantize = quantize
        self._model = None
        self._processor = None
        self._is_loaded = False
        self._is_available = TRANSFORMERS_AVAILABLE
        
        if not self._is_available:
            self._logger.warning(
                "Transformers not available. Install with: pip install torch transformers"
            )
    
    def load(self) -> bool:
        """Load SmolVLA model and processor."""
        if not self._is_available:
            self._logger.error("Cannot load SmolVLA: transformers not installed")
            return False
        
        try:
            self._logger.info(f"Loading SmolVLA model: {self._model_name}")
            
            dtype = torch.float16 if self._quantize else torch.float32
            
            self._processor = AutoProcessor.from_pretrained(
                self._model_name,
                trust_remote_code=True
            )
            
            self._model = AutoModelForVision2Seq.from_pretrained(
                self._model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            
            if self._device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            
            self._model.eval()
            
            self._is_loaded = True
            self._logger.info("SmolVLA loaded successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load SmolVLA: {e}")
            self._is_loaded = False
            return False
    
    def predict_action(
        self,
        image: np.ndarray,
        instruction: str,
        robot_state: Optional[np.ndarray] = None,
    ) -> Optional[VLAAction]:
        """Predict robot action from image and instruction.
        
        Args:
            image: Input image (H, W, 3) BGR format
            instruction: Natural language instruction (e.g., "pick up the cup")
            robot_state: Optional robot joint positions (7 values)
            
        Returns:
            VLAAction with predicted action or None
        """
        if not self._is_loaded:
            return None
        
        try:
            image_pil = Image.fromarray(image)
            
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            
            inputs = self._processor(
                text=prompt,
                images=image_pil,
                return_tensors="pt"
            )
            
            if self._device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )
            
            generated = self._processor.decode(outputs[0], skip_special_tokens=True)
            
            action = self._parse_action(generated, instruction)
            
            return action
            
        except Exception as e:
            self._logger.warning(f"SmolVLA prediction error: {e}")
            return None
    
    def _parse_action(self, generated: str, instruction: str) -> VLAAction:
        """Parse generated text into action.
        
        SmolVLA generates action tokens that need to be mapped to robot commands.
        """
        generated_lower = generated.lower()
        
        action_type = "stop"
        confidence = 0.5
        values = np.zeros(7)
        
        if "forward" in generated_lower or "move" in generated_lower:
            action_type = "move_forward"
            values[0] = 0.1
            confidence = 0.7
        elif "back" in generated_lower:
            action_type = "move_backward"
            values[0] = -0.1
            confidence = 0.7
        elif "left" in generated_lower:
            action_type = "turn_left"
            values[5] = 0.1
            confidence = 0.7
        elif "right" in generated_lower:
            action_type = "turn_right"
            values[5] = -0.1
            confidence = 0.7
        elif "stop" in generated_lower or "wait" in generated_lower:
            action_type = "stop"
            confidence = 0.9
        elif "grab" in generated_lower or "pick" in generated_lower:
            action_type = "gripper_close"
            values[6] = 1.0
            confidence = 0.6
        elif "release" in generated_lower or "drop" in generated_lower:
            action_type = "gripper_open"
            values[6] = 0.0
            confidence = 0.6
        
        return VLAAction(
            action_type=action_type,
            values=values,
            confidence=confidence,
            reasoning=f"SmolVLA: {generated[:100]}...",
        )
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @property
    def is_available(self) -> bool:
        return self._is_available
    
    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._is_loaded = False
        self._logger.info("SmolVLA unloaded")


class OpenVLAWrapper:
    """OpenVLA wrapper for OpenEyes.
    
    OpenVLA is a 7B parameter VLA from Stanford/UC Berkeley.
    More powerful but requires more compute.
    
    Model: openvla/openvla-7b
    
    Note: Requires significant GPU memory (~16GB VRAM)
    """
    
    MODEL_NAME = "openvla/openvla-7b"
    
    def __init__(
        self,
        device: str = "cuda",
        quantize: bool = True,
    ):
        self._logger = get_logger(__name__)
        self._device = device
        self._quantize = quantize
        self._model = None
        self._processor = None
        self._is_loaded = False
        self._is_available = TRANSFORMERS_AVAILABLE
        
        if not self._is_available:
            self._logger.warning("Transformers not available")
    
    def load(self) -> bool:
        """Load OpenVLA model."""
        if not self._is_available:
            return False
        
        try:
            self._logger.info(f"Loading OpenVLA: {self.MODEL_NAME}")
            self._logger.warning("OpenVLA requires ~16GB VRAM. Consider using quantized version.")
            
            dtype = torch.float16
            
            self._processor = AutoProcessor.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True
            )
            
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            if self._device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            
            self._model.eval()
            self._is_loaded = True
            self._logger.info("OpenVLA loaded successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load OpenVLA: {e}")
            return False
    
    def predict_action(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> Optional[VLAAction]:
        """Predict action using OpenVLA."""
        if not self._is_loaded:
            return None
        
        try:
            image_pil = Image.fromarray(image)
            
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            
            inputs = self._processor(prompt, image_pil).to(
                self._device, 
                dtype=torch.bfloat16
            )
            
            with torch.no_grad():
                action = self._model.predict_action(
                    **inputs,
                    unnorm_key="bridge_orig",
                    do_sample=False
                )
            
            return VLAAction(
                action_type="vla_action",
                values=action,
                confidence=0.8,
                reasoning="OpenVLA action prediction",
            )
            
        except Exception as e:
            self._logger.warning(f"OpenVLA prediction error: {e}")
            return None
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @property
    def is_available(self) -> bool:
        return self._is_available
    
    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._is_loaded = False


class OctoWrapper:
    """Octo wrapper for OpenEyes.
    
    Octo is an open-source generalist robot policy (~93M params)
    from Stanford.
    
    Model: octo-models/octo-base
    """
    
    MODEL_NAME = "octo-models/octo-base-0.1"
    
    def __init__(self, device: str = "cuda"):
        self._logger = get_logger(__name__)
        self._device = device
        self._model = None
        self._is_loaded = False
        self._is_available = TRANSFORMERS_AVAILABLE
    
    def load(self) -> bool:
        """Load Octo model."""
        if not self._is_available:
            return False
        
        try:
            self._logger.info(f"Loading Octo: {self.MODEL_NAME}")
            
            from transformers import OctoModel, OctoProcessor
            
            self._processor = OctoProcessor.from_pretrained(self.MODEL_NAME)
            self._model = OctoModel.from_pretrained(self.MODEL_NAME)
            
            if self._device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            
            self._model.eval()
            self._is_loaded = True
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load Octo: {e}")
            return False
    
    def predict_action(
        self,
        images: List[np.ndarray],
        task_description: str,
        robot_state: Optional[np.ndarray] = None,
    ) -> Optional[VLAAction]:
        """Predict action using Octo."""
        if not self._is_loaded:
            return None
        
        try:
            images_pil = [Image.fromarray(img) for img in images]
            
            inputs = self._processor(
                task_description,
                images_pil,
                return_tensors="pt"
            )
            
            if self._device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                actions = self._model(**inputs)
            
            return VLAAction(
                action_type="octo_action",
                values=actions.cpu().numpy()[0],
                confidence=0.8,
                reasoning="Octo policy action",
            )
            
        except Exception as e:
            self._logger.warning(f"Octo prediction error: {e}")
            return None
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


def create_vla_model(
    model_type: str = "smolvla",
    device: str = "cuda",
    **kwargs,
) -> Union[SmolVLAWrapper, OpenVLAWrapper, OctoWrapper, None]:
    """Factory function to create VLA model.
    
    Args:
        model_type: "smolvla", "openvla", or "octo"
        device: "cuda" or "cpu"
        **kwargs: Additional model arguments
        
    Returns:
        VLA model wrapper or None
    """
    model_type = model_type.lower()
    
    if model_type == "smolvla":
        model = SmolVLAWrapper(device=device, **kwargs)
    elif model_type == "openvla":
        model = OpenVLAWrapper(device=device, **kwargs)
    elif model_type == "octo":
        model = OctoWrapper(device=device)
    else:
        return None
    
    if model.load():
        return model
    
    return None
