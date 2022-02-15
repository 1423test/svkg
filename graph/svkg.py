import csv
import argparse
import json

from nltk.corpus import wordnet as wn
import torch
import torch.nn as nn
from glove import GloVe
import os
import random
import glob


class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)

#wordnet_id -> sysnet
def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))

#sysnet -> wordnet
def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s

#Building graph from hypernym relations
def getedges(s):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in enumerate(s):
        for v in u.hypernyms():
            j = dic.get(v)
            if j is not None:
                edges.append((i, j))
    return edges


def dense(wnids,edges,sub):
    adjs = {}
    dense_visual = []
    wnids = wnids + sub

    for i in range(len(wnids)):
        adjs[i] = []
    for u, v in edges:
        adjs[u].append(v)  # 边的键值对
    for u, wnid in enumerate(wnids):
        q = [u]  # [32323]
        l = 0
        d = {}
        d[u] = 0  # d:{32323: 0} d[u] = 0
        while l < len(q):
            x = q[l]
            l += 1
            for y in adjs[x]:  # [32323,2602] -> 2602
                if d.get(y) is None:
                    d[y] = d[x] + 1
                    q.append(y)
                q.append(y)
        for x, dis in d.items():
            if wnid in sub:
                dense_visual.append((u, x))
    dense_visual = list(set(edges))
    return  dense_visual

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='datasets/split/imagenet-split.json')
    parser.add_argument('--output', default='svkg.json')
    args = parser.parse_args()

    print('making graph ...')
    #Hierarchy
    xml_wnids = json.load(open('imagenet-xml-wnids.json', 'r'))
    xml_nodes = list(map(getnode, xml_wnids))
    xml_set = set(xml_nodes)

    js = json.load(open(args.input, 'r'))
    train_wnids = js['train']
    test_wnids = js['test']

    key_wnids = train_wnids + test_wnids

    s = list(map(getnode, key_wnids))

    s_set = set(s)
    for u in xml_nodes:
        if u not in s_set:
            s.append(u)

    wnids = list(map(getwnid, s))
    edges = getedges(s)
    n = len(wnids)
    dic = dict(zip(wnids, range(len(wnids))))
    visual = []
    visual_ed = []
    aug_ed = []
    aug = []

    print('Adding visual nodes...')
    dir = 'datasets/part/'
    num = len(glob.glob(r'datasets/part/*.json'))
    for root, dirs, files in os.walk(dir):
        for file in files:
            node_part = os.path.splitext(file)[0]  # node_part
            cls = os.path.splitext(file)[0].split('_')[0]  # node
            node_part_id = len(wnids) + len(visual)
            # visual edges
            visual_ed.append((node_part_id, dic.get(cls)))
            #visual node
            visual.append(node_part)
            # aug edges
            aug_ed.append((len(wnids)+ num,dic.get(cls)))
    dense_visual = dense(wnids,edges+visual_ed,visual)

    print('making semantic and visual embedding ...')
    #生成glove向量
    glove = GloVe('glove.6B.300d.txt')
    vectors = []
    # semantic embedding
    visual_vectors = []
    for wnid in wnids:
        vectors.append(glove[getnode(wnid).lemma_names()].cuda())
    # visual embedding
    for vis in visual:
            v = json.load(open(os.path.join(dir, vis + '.json')))
            v = torch.Tensor(v).cuda()
            visual_vectors.append(v)

    aug.append('aug')
    aug_vector = torch.zeros(1,300)
    vectors = torch.stack(vectors)
    visual_vectors = torch.stack(visual_vectors)

    obj = {}
    print('Semantic: {} nodes, {} edges'.format(len(vectors), len(edges)))
    obj['edges'] = edges
    obj['wnids'] = wnids
    obj['vectors'] = vectors.tolist()

    print('Visual: {} nodes, {} edges'.format(len(visual_vectors), len(dense_visual)))
    obj['visual'] = visual
    obj['visual_ed'] = dense_visual
    obj['visual_vectors'] = visual_vectors.tolist()

    print('Augmentation: {} nodes, {} edges'.format(1, len(aug_ed)))
    obj['aug'] = aug
    obj['aug_vector'] = aug_vector.tolist()
    obj['aug_ed'] = aug_ed


    json.dump(obj, open(args.output, 'w'),cls=MyEncoder)


