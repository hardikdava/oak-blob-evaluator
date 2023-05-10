# Oak-blob-evaluator

This repository is made on top of yolov5 repository. The goal is to evaulate oak-blob model on evaluation dataset to identify errors in model conversion. 

## Model Generation:

- Train your model (yolov5, yolov6, yolov7) using their framework.
- Save custom weights to pytorch model 
- Generate blob model using [DepthAI-Tools](https://tools.luxonis.com/)


## Usage:

```python val.py --weights {blob_model_directory} --imgsz {416} --name {save_dir_name} --project {runs/val}```

#### Notes:
- blob_model_directory should have **.blob model** and metadata **json file**.
- At least a one depthai device (can be any model) should be attached to host.


### References:

```
@software{glenn_jocher_2020_4154370,
  author       = {Glenn Jocher and,Alex Stoken and,Jirka Borovec and,NanoCode012 and,ChristopherSTAN and,Liu Changyu and,Laughing and,tkianai and,Adam Hogan and,lorenzomammana and,yxNONG and,AlexWang1900 and,Laurentiu Diaconu and,Marc and,wanghaoyang0106 and,ml5ah and,Doug and,Francisco Ingham and,Frederik and,Guilhen and,Hatovix and,Jake Poznanski and,Jiacong Fang and,Lijun Yu 于力军 and,changyu98 and,Mingyu Wang and,Naman Gupta and,Osama Akhtar and,PetrDvoracek and,Prashant Rai},
  title={{ultralytics/yolov5: v7.2 - Bug Fixes and 
                   Performance Improvements}},
  month= oct,
  year= 2020,
  publisher= {Zenodo},
  version= {v3.1},
  doi= {10.5281/zenodo.4154370},
  url= {https://doi.org/10.5281/zenodo.4154370}
}

```

