import os
import time
import numpy as np
import tensorrt as trt
import pyds


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTDetector:
    def __init__(self, engine_path, conf_thresh=0.25, nms_thresh=0.45):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.engine_path = engine_path
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        self.input_idx = self.engine.get_binding_index("images")
        self.output_idx = self.engine.get_binding_index("output")
        
        self.input_shape = self.engine.get_binding_shape(self.input_idx)
        self.output_shape = self.engine.get_binding_shape(self.output_idx)
        
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        
        self.labels = self._load_labels()
        
    def _load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(engine_data)
    
    def _load_labels(self):
        labels_path = "/home/mandar/openeyes/deepstream/labels.txt"
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        return [f"class_{i}" for i in range(80)]
    
    def _preprocess(self, img):
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img
    
    def _postprocess(self, outputs, img_shape):
        detections = []
        
        outputs = outputs[0]
        
        if len(outputs.shape) == 2:
            for detection in outputs:
                x1, y1, x2, y2, conf, cls = detection[:6]
                
                if conf < self.conf_thresh:
                    continue
                
                cls = int(cls)
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': cls,
                    'label': self.labels[cls] if cls < len(self.labels) else f"class_{cls}"
                })
        
        return detections
    
    def detect(self, img):
        img_resized = self._resize_letterbox(img)
        img_input = self._preprocess(img_resized)
        
        d_input = np.ascontiguousarray(img_input)
        d_output = np.empty(self.output_shape, dtype=np.float32)
        
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        h_input = cuda.mem_alloc(d_input.nbytes)
        h_output = cuda.mem_alloc(d_output.nbytes)
        
        stream = cuda.Stream()
        
        cuda.memcpy_htod(h_input, d_input)
        self.context.execute_v2([int(h_input), int(h_output)])
        cuda.memcpy_dtoh(d_output, h_output)
        
        detections = self._postprocess(d_output, img.shape[:2])
        
        h_input.free()
        h_output.free()
        
        return detections
    
    def _resize_letterbox(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        import cv2
        resized = cv2.resize(img, (new_w, new_h))
        
        padded = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        top = (self.input_h - new_h) // 2
        left = (self.input_w - new_w) // 2
        padded[top:top+new_h, left:left+new_w] = resized
        
        return padded


def create_tensorrt_detector():
    engine_path = "/home/mandar/openeyes/models/yolov10n.engine"
    
    if not os.path.exists(engine_path):
        print(f"Warning: TensorRT engine not found at {engine_path}")
        print("Falling back to ONNX inference with TensorRT runtime...")
        return None
    
    try:
        return TensorRTDetector(engine_path)
    except Exception as e:
        print(f"Error loading TensorRT engine: {e}")
        return None
