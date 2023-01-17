import glob
import numpy as np
import torch
from utils.general import (LOGGER, check_requirements)


class DetectMultiBackend:
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), oak=True):

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        if oak:
            blob = True
            nhwc = True  # BHWC formats (vs torch BCWH)
            LOGGER.info(f'Using {w} as Inference Server...')
            check_requirements('depthai')
            from models.oak_inference import OakInference
            configPath = glob.glob(str(w) + "/*.json")[0]
            nnPath = glob.glob(str(w) + "/*.blob")[0]
            model = OakInference(configPath=configPath, nnPath=nnPath, image_size=416)
            names = model.labels

        self.__dict__.update(locals())  # assign all variables to self

    def infer(self, im):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.blob:
            im = im.cpu().numpy()[0]
            y = self.model.infer(im)
            y = np.expand_dims(y, 0)

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)
    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x