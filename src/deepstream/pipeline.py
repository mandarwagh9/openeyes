import sys
import os

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import pyds


class DeepStreamPipeline:
    def __init__(self, config_file, camera_source=0):
        self.config_file = config_file
        self.camera_source = camera_source
        self.pipeline = None
        self.loop = None
        self.running = False
        self.detections = []

        Gst.init(None)

    def create_camera_source(self):
        return (
            f"nvarguscamerasrc sensor-id={self.camera_source} ! "
            "video/x-raw(memory:NVMM),width=640,height=480,format=NV12,framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=appsink emit-signals=True drop=True"
        )

    def create_pipeline(self):
        camera_src = self.create_camera_source()

        self.pipeline = Gst.parse_launch(
            f"{camera_src} ! queue ! appsink.sink_0"
        )

        appsink = self.pipeline.get_by_name("appsink")
        appsink.connect("new-sample", self.on_new_sample)

    def on_new_sample(self, appsink):
        sample = appsink.pop_sample()
        if not sample:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        if not buf:
            return Gst.FlowReturn.OK

        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = struct.get_int("width").value
        height = struct.get_int("height").value

        _, info = buf.map(Gst.MapFlags.READ)
        import numpy as np
        frame = np.ndarray(
            (height * width * 3,), 
            buffer=info.data, 
            dtype=np.uint8
        ).reshape((height, width, 3))
        buf.unmap(info)

        self.process_frame(frame)

        return Gst.FlowReturn.OK

    def process_frame(self, frame):
        pass

    def run(self):
        self.loop = GLib.MainLoop()
        self.running = True

        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            self.loop.run()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop:
            self.loop.quit()


class DeepStreamYOLOPipeline(DeepStreamPipeline):
    def __init__(self, config_file, camera_source=0):
        super().__init__(config_file, camera_source)
        self.config_file = config_file
        
    def create_pipeline(self):
        pipeline_str = (
            f"nvarguscamerasrc sensor-id={self.camera_source} ! "
            "video/x-raw(memory:NVMM),width=640,height=480,format=NV12,framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "queue ! "
            "muxer.sink_0 "
            "nvstreammux name=muxer batch-size=1 width=640 height=480 ! "
            "nvinfer name=primary-nvinference config-file-path={self.config_file} ! "
            "queue ! "
            "nvdsosd name=nvdsosd ! "
            "queue ! "
            "nvvidconv flip-method=0 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=appsink emit-signals=True drop=True"
        )
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        appsink = self.pipeline.get_by_name("appsink")
        appsink.connect("new-sample", self.on_new_sample)
        
        osd = self.pipeline.get_by_name("nvdsosd")
        osd.set_property('process-mode', 0)
        osd.set_property('display-text', True)
        
    def process_frame(self, frame):
        pass


def run_deepstream_inference(config_file, camera_source=0):
    pipeline = DeepStreamYOLOPipeline(config_file, camera_source)
    pipeline.create_pipeline()
    pipeline.run()


if __name__ == "__main__":
    config = "/home/mandar/openeyes/deepstream/config_yolov10.txt"
    run_deepstream_inference(config, 0)
