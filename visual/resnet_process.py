from PIL import Image, ImageFile
import os, torch, glob
import numpy as np
from torch.autograd import Variable
from torchvision import models, transforms
from resnet import make_resnet50_base
import json
from torchvision import get_image_backend
import torch.nn as nn
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

# get folder name
def filename_get(data_dir):
    L = []
    files=os.listdir(data_dir)
    for f in files:
        if os.path.isdir(data_dir + '/'+f):
            L.append(f)
    return L


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def pil_loader(path):
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
        except OSError:
            return None
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def extractor(img_path,cnn,use_gpu):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = default_loader(img_path)

    i = 0
    for i in range(0,1):
        if image == None:
            continue
        else:
            # print(' From image file {}'.format(image) + " " + img_path)
            img = preprocess(image)
            x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
            if use_gpu:
                x = x.cuda()
                net = cnn.cuda()
                y = net(x).cpu().flatten()
                y = y.data.numpy()
        return y

if __name__ == '__main__':
    path = 'D:\imagenet'
    extensions = ['jpeg']
    use_gpu = torch.cuda.is_available()

    i = 0
    files_list = []

    print("loading model")
    cnn = make_resnet50_base()
    cnn.load_state_dict(torch.load('resnet50-base.pth'))
    cnn = cnn.cuda()
    cnn.eval()

    sub_dirs = []
    print("acquire file path")
    sub_dirs = filename_get(path)
    for root,dirs,files in os.walk(path):
        for file in files:
            sub_dirs.append(os.path.splitext(file)[0])
    print(sub_dirs)


    print("extract feature")
    i = 0
    for sub_dir in sub_dirs:
        class_name  = sub_dir.split('/')[1]
        sub_dir = os.path.join(path,sub_dir)

        for extention in extensions:
            file_glob = os.path.join(sub_dir, '*.' + extention)
            files_list = glob.glob(file_glob)
            random.shuffle(files_list)
            files_list = files_list[:max(1, round(len(files_list)))]
            if files_list == []:
                continue
            else:
                vectors = []
                for filename in files_list:
                    feature = extractor(filename, cnn, use_gpu)
                    vectors.append(feature)
                i = i + 1
                print(str(i) + ' ' + class_name +'images number：' + str(len(vectors)))
                dir  = 'datasets/imagenet/'+ class_name +'.json'
                json.dump(vectors, open(dir, 'w', encoding='utf-8'), cls=NpEncoder)

