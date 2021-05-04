# HyperPoint: SuperPoint Loss Improvement

This is a PyTorch implementation of "SuperPoint: Self-Supervised Interest Point Detection and Description." Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich. ArXiv 2018. This code is partially based on the tensorflow implementation https://github.com/rpautrat/SuperPoint and pytorch implementation https://github.com/eric-yyjau/pytorch-superpoint

## Differences between our implementation and original paper
- *Descriptor loss*: We tested descriptor loss using different methods, including dense method (as paper but slightly different) and sparse method. We notice sparse loss can converge more efficiently with similar performance. The default setting here is sparse method.

## Results on HPatches
| Task                                      | Homography estimation |      |      | Detector metric |      | Descriptor metric |                |
|-------------------------------------------|-----------------------|------|------|-----------------|------|-------------------|----------------|
|                                           | Epsilon = 1           | 3    | 5    | Repeatability   | MLE  | NN mAP            | Matching Score |
| Pretrained model                        | 0.44                  | 0.77 | 0.83 | 0.606           | 1.14 | 0.81              | 0.55           |
| Sift (subpixel accuracy)                  | 0.63                  | 0.76 | 0.79 | 0.51            | 1.16 | 0.70               | 0.27            |
| superpoint_coco_heat2_0_170k_hpatches_sub | 0.46                  | 0.75 | 0.81 | 0.63            | 1.07 | 0.78              | 0.42           |
| superpoint_kitti_heat2_0_50k_hpatches_sub | 0.44                  | 0.71 | 0.77 | 0.56            | 0.95 | 0.78              | 0.41           |

- Pretrained model is from [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).
- The evaluation is done under our evaluation scripts.
- COCO/ KITTI pretrained model is included in this repo.


## Installation
### Requirements
- python == 3.6
- pytorch >= 1.1 (tested in 1.3.1)
- torchvision >= 0.3.0 (tested in 0.4.2)
- cuda (tested in cuda10)

```
conda create --name py36-sp python=3.6
conda activate py36-sp
pip install -r requirements.txt
pip install -r requirements_torch.txt # install pytorch
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip install opencv-contrib-python
```

### Path setting
- paths for datasets ($DATA_DIR), logs are set in `setting.py`

### Dataset
Datasets should be downloaded into $DATA_DIR. The Synthetic Shapes dataset will also be generated there. The folder structure should look like:

```
datasets/ ($DATA_DIR)
|-- COCO
|   |-- train2014
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2014
|       |-- file1.jpg
|       `-- ...
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
`-- synthetic_shapes  # will be automatically created

- MS-COCO 2014 
    - [MS-COCO 2014 link](http://cocodataset.org/#download)
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)

```
## run the code
- Notes:
    - Start from any steps (1-4) by downloading some intermediate results
    - Training usually takes 8-10 hours on one 'NVIDIA 2080Ti'.
    - Currently Support training on 'COCO' dataset (original paper), 'KITTI' dataset.
- Tensorboard:
    - log files is saved under 'runs/<\export_task>/...'
    
`tensorboard --logdir=./runs/ [--host | static_ip_address] [--port | 6008]`

### 1) Training Superpoint on MS-COCO
You need pseudo ground truth labels to traing detectors. Labels can be exported from step 2) or downloaded from [link](https://drive.google.com/drive/folders/1nnn0UbNMFF45nov90PJNnubDyinm2f26?usp=sharing). Then, as usual, you need to set config file before training.
- config file
  - root: specify your labels root
  - root_split_txt: where you put the train.txt/ val.txt split files (no need for COCO, needed for KITTI)
  - labels: the exported labels from homography adaptation
  - pretrained: specify the pretrained model (you can train from scratch)
- 'eval': turn on the evaluation during training 

#### General command
```
python train4.py <train task> <config file> <export folder> --eval
```

#### COCO
```
python train4.py train_joint configs/superpoint_coco_train_heatmap.yaml superpoint_coco --eval --debug
```

- set your batch size (originally 1)
- refer to: 'train_tutorial.md'

### 2) Export/ Evaluate the metrics on HPatches
- Use pretrained model or specify your model in config file
- ```./run_export.sh``` will run export then evaluation.

#### Export
- download HPatches dataset (link above). Put in the $DATA_DIR.
```python export.py <export task> <config file> <export folder>```
- Export keypoints, descriptors, matching
```
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap.yaml superpoint_hpatches_test
```
#### evaluate
```python evaluation.py <path to npz files> [-r, --repeatibility | -o, --outputImg | -homo, --homography ]```
- Evaluate homography estimation/ repeatability/ matching scores ...
```
python evaluation.py logs/superpoint_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching
```



- specify the pretrained model

## Pretrained models
### Current best model
- *COCO dataset*
```logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar```

### model from magicleap
```pretrained/superpoint_v1.pth```

## Jupyter notebook 
```shell
# show images saved in the folders
jupyter notebook
notebooks/visualize_hpatches.ipynb 
```


## Known problems
- Export COCO dataset in low resolution (240x320) instead of high resolution (480x640).
- Due to step 1 was done long time ago. We are still testing it again along with step 2-4. Please refer to our pretrained model or exported labels. Or let us know how the whole pipeline works.
- Warnings from tensorboard.

## Citations
Please cite the original paper.
```
@inproceedings{detone2018superpoint,
  title={Superpoint: Self-supervised interest point detection and description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={224--236},
  year={2018}
}
```



# Credits
This implementation is developed by [Sayantan Chattopadhyay] (https://github.com/schattop6), [Jordan Hart] (https://github.com/jordanhart) and [Sanjay Rathore]. Please contact Sayantan Chattopadhyay for any problems. 
Again the work is based on Tensorflow implementation by [Rémi Pautrat](https://github.com/rpautrat) and [Paul-Edouard Sarlin](https://github.com/Skydes) and official [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork) and pytorch implementation by [You-Yi Jau] (https://github.com/eric-yyjau/pytorch-superpoint) 

