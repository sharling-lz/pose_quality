# Learning to Acquire the Quality of Human Pose Estimation

## Introduction
This is an official pytorch implementation of [*Learning to Acquire the Quality of Human Pose Estimation*]. 
In this work, we propose end-to-end human pose quality learning, which adds a quality prediction block alongside pose regression. The proposed block learns the object keypoint similarity (OKS) between the estimated pose and its corresponding ground truth by sharing the pose features with heatmap regression. Utilizing the learned quality as pose score improves pose estimation performance during COCO AP evaluation.</br>

<img src="/figures/oks-net.jpg" width = "400" alt="Illustrating the architecture of the proposed OKS-Net" align=center />

## Main Results
### Results on COCO val2017 with ground truth bounding box
| Arch               | Input size |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|------------|-------|-------|--------|--------|--------|
|   pose_hrnet_w32   |    256x192 | 0.765 | 0.935 |  0.837 |  0.739 |  0.808 |
|         +OKS-net   |            | 0.775 | 0.937 |  0.850 |  0.747 |  0.820 |
|   pose_hrnet_w32   |    384x288 | 0.777 | 0.936 |  0.847 |  0.748 |  0.825 |
|         +OKS-net   |            | 0.785 | 0.936 |  0.851 |  0.753 |  0.838 |
|   pose_hrnet_w48   |    256x192 | 0.771 | 0.936 |  0.847 |  0.741 |  0.819 |
|         +OKS-net   |            | 0.777 | 0.937 |  0.850 |  0.748 |  0.829 |
|   pose_hrnet_w48   |    384x288 | 0.781 | 0.936 |  0.849 |  0.753 |  0.831 |
|         +OKS-net   |            | 0.787 | 0.936 |  0.849 |  0.753 |  0.841 |

### Note:
- Flip test is used.
- The results are obtained using pytorch 1.0, minor differences may be get using higher pytorch versions.

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 2 NVIDIA 2080ti GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

6. Download pretrained models from our model zoo([GoogleDrive])(https://drive.google.com/drive/folders/1Fxpn-phF3M7TStuxDqfNJ0Bpdb1JwHJP?usp=sharing)
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- pose_coco
            |   |-- pose_hrnet_w32_256x192.pth
            |   |-- pose_hrnet_w32_384x288.pth
            |   |-- pose_hrnet_w48_256x192.pth
            |   |-- pose_hrnet_w48_384x288.pth
            `-- pose_quality_coco
                |-- pose_quality_hrnet_w32_256x192.pth
                |-- pose_quality_hrnet_w32_384x288.pth
                |-- pose_quality_hrnet_w48_256x192.pth
                |-- pose_quality_hrnet_w48_384x288.pth

   ```
   
### Data preparation
**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing
#### Testing on COCO val2017 dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1Fxpn-phF3M7TStuxDqfNJ0Bpdb1JwHJP?usp=sharing) )
 

```
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_quality_coco/pose_quality_hrnet_w32_256x192.pth
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
```

### Citation
If you use our code or models in your research, please cite with:
```
@ARTICLE{zhao_learning_2021,
  author={Zhao, Lin and Xu, Jie and Gong, Chen and Yang, Jian and Zuo, Wangmeng and Gao, Xinbo},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Learning to Acquire the Quality of Human Pose Estimation}, 
  year={2021},
  volume={31},
  number={4},
  pages={1555-1568},
  doi={10.1109/TCSVT.2020.3005522}}
```

### Acknowledgement
The codes are developed based on the opensource of [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation).
