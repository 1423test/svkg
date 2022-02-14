## Requirements

* python 3.7
* pytorch 1.6.0
* nltk 3.4

### Data Preparation
#### ImageNet and part visual features 

1. Download ImageNet.
2. Download part visual features:https://drive.google.com/drive/folders/19Hg59bfIusNLNLONkxoswPbsCUGpb_Gd?usp=sharing and  `cd datasets/part`

An ImageNet root directory should contain image folders, each folder with the wordnet id of the class.

#### Glove Word Embedding
1. Download: http://nlp.stanford.edu/data/glove.6B.zip
2. Unzip it, find and put `glove.6B.300d.txt` to `graph/`.

#### Graph
1. `cd graph/`
2. Run `svkg.py`, get `svkg.json`
3. Else download svkg.json from https://drive.google.com/drive/folders/1zcrr3d7UYZBF5X7hQvG_bR0R3xFehGiC?usp=sharing

#### Pretrained ResNet50
1. Download: https://drive.google.com/drive/folders/1hZ0CcAs8UwO9YsnNNs7Ve7i9AtSLfL4N?usp=sharing
2. Put files :`cd visual/fc-weights.json` and `visual/resnet50-base.pth`
3. Run 'python resnet_process.py' get visual features of imagenet in 'datasets/imagenet'

#### Train Graph Networks
Run `python train.py`, get results in `baseline/svkg-1000.pred`
*(Download pretrained model from https://drive.google.com/drive/folders/1zcrr3d7UYZBF5X7hQvG_bR0R3xFehGiC?usp=sharing for testing)

### Testing
Run `python test.py` with the args:

#### ImageNet
* `--pred`: the `.pred` file for testing. 
* `--test-set`: choose test set, choices: `[general, detailed]`
* `--split`: choose test set, choices: `[2-hops, 3-hops, all, bird, dog, snake, monkey]`
* (optional) `--keep-ratio` for the ratio of testing data, `--consider-trains` to include training classes' classifiers, `--test-train` for testing with train classes images only.

