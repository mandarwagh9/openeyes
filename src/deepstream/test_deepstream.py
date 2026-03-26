#!/usr/bin/env python3
"""
DeepStream YOLO Pipeline Test
Tests the DeepStream pipeline with YOLOv10 for object detection.
"""

import sys
import os

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import cv2


class DeepStreamYOLOTest:
    def __init__(self):
        Gst.init(None)
        self.loop = GLib.MainLoop()
        self.running = False
        self.frame_count = 0
        self.fps_start = 0
        
    def create_pipeline(self):
        pipeline_str = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM),width=640,height=480,format=NV12,framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=appsink emit-signals=True max-buffers=1 drop=True"
        )
        
        print("Creating pipeline...")
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        appsink = self.pipeline.get_by_name("appsink")
        appsink.connect("new-sample", self.on_new_sample)
        
        return self.pipeline
    
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
        
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK
            
        try:
            frame = np.ndarray(
                (height * width * 3,),
                buffer=map_info.data,
                dtype=np.uint8
            ).reshape((height, width, 3))
            
            self.frame_count += 1
            if self.frame_count == 1:
                self.fps_start = cv2.getTickCount()
            
            if self.frame_count % 30 == 0:
                fps = (self.frame_count - 1) / ((cv2.getTickCount() - self.fps_start) / cv2.getTickFrequency())
                print(f"FPS: {fps:.2f}")
                cv2.imwrite(f"/tmp/frame_{self.frame_count}.jpg", frame)
                
        finally:
            buf.unmap(map_info)
            
        return Gst.FlowReturn.OK
    
    def run(self):
        self.create_pipeline()
        
        print("Starting pipeline...")
        self.pipeline.set_state(Gst.State.PLAYING)
        
        self.running = True
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop()
    
    def stop(self):
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop:
            self.loop.quit()
        print(f"Total frames: {self.frame_count}")


def main():
    print("=" * 50)
    print("DeepStream YOLO Pipeline Test")
    print("=" * 50)
    
    tester = DeepStreamYOLOTest()
    tester.run()


if __name__ == "__main__":
    main()
