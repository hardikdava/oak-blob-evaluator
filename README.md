# oak-blob-evaluator


This resposity is pre evaluation step which should be done before updating model for deployment. It is developed on top of YoloV5 repository. Currently, only Pytorch TXT format is support. This is 100% working for YoloV5, YoloV7, and YoloV8 models.

## Usage:

```python val.py --weights {blob_model_directory} --imgsz {416} --name {save_dir_name} --project {runs/val}```

blob_model_directory should have .blob model and metadata json file.

### TODO:

- [ ] Custom loading in oak_inference
- [ ] Support for other model evaluation i.e. onnx: might be useful for TensorRT (Jetson deployment)
- [ ] Other dataset support.
- [ ] pytorch model support using HUB
