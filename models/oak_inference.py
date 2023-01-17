#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import json
from time import monotonic
import glob
from utils.general import LOGGER
import torch


class OakInference:

    def __init__(self, configPath, nnPath, image_size):
        self.labels = None
        self.config = dict()
        self.configPath = configPath
        self.nnPath = nnPath
        self.image_size = image_size



        self.pipeline = self.create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.device.startPipeline()
        self.qDet = self.device.getOutputQueue(name="nnOut", maxSize=4, blocking=False)
        self.qNnIn = self.device.getInputQueue(name="nnIn")


    def __call__(self, img):
        return self.infer(img)


    def create_pipeline(self):
        with open(self.configPath) as f:
            self.config = json.load(f)
        nnConfig = self.config.get("nn_config", {})

        # parse input shape
        if "input_size" in nnConfig:
            self.W, self.H = tuple(map(int, nnConfig.get("input_size").split('x')))

        # extract metadata
        metadata = nnConfig.get("NN_specific_metadata", {})
        self.classes = int(metadata.get("classes", {}))
        self.coordinates = metadata.get("coordinates", {})
        self.anchors = metadata.get("anchors", {})
        self.anchorMasks = metadata.get("anchor_masks", {})
        self.iouThreshold = metadata.get("iou_threshold", {})
        self.confidenceThreshold = metadata.get("confidence_threshold", {})

        # self.confidenceThreshold = 0.001  ## Override confidence threshold

        # parse labels
        self.nnMappings = self.config.get("mappings", {})
        self.labels = self.nnMappings.get("labels", {})
        # sync outputs

        # Create pipeline
        pipeline = dai.Pipeline()
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)

        nnOut = pipeline.create(dai.node.XLinkOut)
        nnIn = pipeline.create(dai.node.XLinkIn)

        nnOut.setStreamName("nnOut")
        nnIn.setStreamName("nnIn")

        # Network specific settings
        detectionNetwork.setConfidenceThreshold(self.confidenceThreshold)
        detectionNetwork.setNumClasses(self.classes)
        detectionNetwork.setCoordinateSize(self.coordinates)
        detectionNetwork.setAnchors(self.anchors)
        detectionNetwork.setAnchorMasks(self.anchorMasks)
        detectionNetwork.setIouThreshold(self.iouThreshold)
        detectionNetwork.setBlobPath(self.nnPath)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(True)

        nnIn.out.link(detectionNetwork.input)
        detectionNetwork.out.link(nnOut.input)

        return pipeline

    def infer(self, img):
        results = np.array([])
        dai_img = dai.ImgFrame()
        in_img = cv2.resize(img, (self.W, self.H))
        in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
        arr = in_img.transpose(2, 0, 1).flatten()
        dai_img.setData(arr)
        dai_img.setTimestamp(monotonic())
        dai_img.setWidth(self.W)
        dai_img.setHeight(self.W)
        self.qNnIn.send(dai_img)  ## sen
        inDet = self.qDet.get()
        if inDet is not None:
            detections = inDet.detections
            if detections:
                results = self.scale_boxes(img, detections)
        return results

    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[0]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def scale_boxes(self, orig_frame, detections: dai.ImgDetections):
        results = list()
        for detection in detections:
            bbox = self.frameNorm(orig_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            results.append([bbox[0], bbox[1], bbox[2], bbox[3], detection.confidence, detection.label])
        results = np.asarray(results)
        return results

    def draw_detections(self, frame: np.ndarray, detections: np.ndarray):
        color = (255, 0, 0)
        for det in detections:
            if det[-1]> 0.5:
                cv2.putText(frame, str(self.labels[int(det[5])]), (int(det[0]) + 10, int(det[1]) + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), color, 2)
        return frame

if __name__ == "__main__":
    configPath = "../v7_blob/metadata.json"
    nnPath = "../v7_blob/model.blob"
    model = OakInference(configPath, nnPath, 416)
    img = cv2.imread("../../datasets/coco128/images/train2017/000000000025.jpg")
    detections = model.infer(img)