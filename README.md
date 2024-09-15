# IEEE BigData Cup 2024: Building Extraction

This repository contains the source code for the competition scheduled to be held in IEEE BigData conference in 2024, [IEEE BigData Cup 2024: Building Extraction](https://www.kaggle.com/competitions/building-extraction-generalization-2024). Our model secured the second position in the competition.

The source code is based on the [MMDetection framework](https://github.com/open-mmlab/mmdetection). We will provide instructions on how to train a model for the building extraction competition and test the trained model.

## Installation

Please follow [the instruction of MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html) and install MMCV and MMEngine. After installing them, clone this repository, go to the cloned directory, and install this framework as follows:
```bash
$ pip install -v -e .
```

## Data preparation

### Image and annotations

To follow the COCO annotation format, we slightly changed the annotations provided from the competition organizer and placed the annotation files in `data/annotaitons`. Please download the competition dataset from [here]() and place the images in the `data/images` directory. The directory names must be changed as shown in the folder structure below.

We also used an external dataset, namely [the Alabama dataset](). We placed the annotation data for the Alabama dataset in the `data/annotations` directory. Please download the dataset and place the images as shown in the folder structure below.

To align with the COCO annotation format, we made slight modifications to the annotations provided by the competition organizer and saved them in the `data/annotations` directory. You can download the competition dataset from [this link](https://www.kaggle.com/competitions/building-extraction-generalization-2024/data) and store the images in the `data/images` directory. Ensure that the directory names match the structure shown below.

Additionally, we used an external dataset, [the Alabama Buildings Segmentation dataset](https://www.kaggle.com/datasets/meowmeowplus/alabama-buildings-segmentation). The annotation file for the Alabama dataset is placed in the `data/annotations` directory. Download the dataset and arrange the images according to the folder structure below.

```
.
├── data
:   ├── images
    │     ├── bigdata_train
    │     │     ├── CBD_0001_0_0.jpg
    │     │     ├── CBD_0001_0_1.jpg
    │     │     :
    │     ├── bigdata_val
    │     ├── bigdata_test
    │     └── alabama
    └── annotations
          ├── bigdata_train.json
          ├── bigdata_val.json
          ├── bigdata_test.json
          └── alabama.json
```

### Pretrained model

The model is initialized with the COCO pretrained model. Please download the pretrained parameters from [this link](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_x_8xb16-300e_coco/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth) and store it in the top of the directory as follows:
```
.
├── configs
├── convert_to_submit.py
:
├── rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth
: 
```

## Training

The training can be started with the following commands.

Training with a single GPU.
```bash
$ python tools/train.py configs/building/rtmdet-ins_x_allaug_alabama_val.py
```

Training with 4 GPUs.
```bash
$ ./tools/dist_train.sh configs/building/rtmdet-ins_x_allaug_alabama_val.py 4
```

The output data will be stored in the `work_dirs/rtmdet-ins_x_allaug_alabama_val` directory.

## Testing

After training is complete, you can evaluate the model using the test set. The following command runs the testing process on the competition test set and generates the results in a `dets.pkl` file.

```bash
$ python tools/test.py \
      work_dirs/rtmdet-ins_x_allaug_alabama_val/rtmdet-ins_x_allaug_alabama_val.py \
      work_dirs/rtmdet-ins_x_allaug_alabama_val/epoch_24.pth \
      --work-dir pred_test \
      --out pred_test/dets.pkl
```

To submit your results to the competition, the output pickle file must be converted to a CSV format. Use the command below to convert the `dets.pkl` file to a CSV file. By default, the output file will be named `dets.csv`.

```
python convert_to_submit.py --input_path pred_test/dets.pkl
```