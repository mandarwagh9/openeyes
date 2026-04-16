// Custom parser for YOLOv10 output
// Output format: [1, 300, 6] = [batch, detections, (x,y,w,h,conf,class)]

#include <cmath>
#include <iostream>
#include <vector>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseCustomYoloV10(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Find the output layer
    NvDsInferLayerInfo const* layer = nullptr;
    for (auto& l : outputLayersInfo) {
        if (l.layerName && strstr(l.layerName, "output")) {
            layer = &l;
            break;
        }
    }
    
    if (!layer || !layer->buffer) {
        std::cerr << "Output layer not found" << std::endl;
        return false;
    }
    
    float* output = (float*)layer->buffer;
    
    // Output shape: [1, 300, 6]
    int numDetections = 300;
    int numClasses = detectionParams.numClassesInstalled;
    float threshold = detectionParams.preClusterThreshold;
    
    for (int i = 0; i < numDetections; i++) {
        float* det = output + i * 6;
        
        float x = det[0];
        float y = det[1];
        float w = det[2];
        float h = det[3];
        float conf = det[4];
        float classId = det[5];
        
        // Filter by confidence
        if (conf < threshold) continue;
        
        // Clamp class ID
        int clsId = (int)classId;
        if (clsId >= numClasses) clsId = 0;
        
        NvDsInferParseObjectInfo obj;
        obj.classId = clsId;
        obj.confidence = conf;
        
        // Convert from center format to corner format
        obj.left = x - w/2;
        obj.top = y - h/2;
        obj.width = w;
        obj.height = h;
        
        objectList.push_back(obj);
    }
    
    return true;
}
