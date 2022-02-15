import argparse
import json
import os.path as osp
import os

import torch
from torch.utils.data import DataLoader

from dataloader import ImageNet
from utils import set_gpu, pick_vectors

def test_on_subset(dataset,n, pred_vectors, all_label,
                   consider_trains):
    top = [1, 2, 5, 10, 20]
    hits = torch.zeros(len(top)).cuda()
    tot = 0

    loader = DataLoader(dataset=dataset, batch_size=1024,
                        shuffle=False, num_workers=8)

    for batch_id, batch in enumerate(loader, 1):
        data, label = batch
        data = data.cuda()

        feat = torch.cat([data, torch.ones(len(data)).view(-1, 1).cuda()], dim=1)
        fcs = pred_vectors.t()

        table = torch.matmul(feat, fcs)
        if not consider_trains:
            table[:, :n] = -1e18

        gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
        rks = (table >= gth_score).sum(dim=1)

        assert (table[:, all_label] == gth_score[:, all_label]).min() == 1

        for i, k in enumerate(top):
            hits[i] += (rks <= k).sum().item()
        tot += len(data)

    return hits, tot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred',default='part/map/baseline/svkg-1000.pred')

    parser.add_argument('--test-set',default='general',help='general, detailed')
    parser.add_argument('--split', default='2-hops', help='2-hops, 3-hops, all, bird, dog, snake, monkey')

    parser.add_argument('--output', default=None)

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--keep-ratio', type=float, default=0.1)
    parser.add_argument('--consider-trains', action='store_true',default=True)

    args = parser.parse_args()

    set_gpu(args.gpu)

    if args.test_set == 'general':
        test_sets = json.load(open('datasets/split/imagenet-testsets.json', 'r'))
        train_wnids = test_sets['train']
        test_wnids = test_sets[args.split]
    elif args.test_set == 'detailed':
        test_sets = json.load(open('datasets/split/'+ args.split +'-split.json', 'r'))
        train_wnids = test_sets['train']
        test_wnids = test_sets['test']

    print('test set: {}, {} classes, ratio={}'
          .format(args.test_set, len(test_wnids), args.keep_ratio))
    print('consider train classifiers: {}'.format(args.consider_trains))

    pred_file = torch.load(args.pred)
    pred_wnids = pred_file['wnids']
    pred_vectors = pred_file['pred']
    pred_dic = dict(zip(pred_wnids, pred_vectors))
    pred_vectors = pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True).cuda()

    pred_vectors = pred_vectors.cuda()

    n = len(train_wnids)
    m = len(test_wnids)


    imagenet_path = 'datasets/imagenet'
    dataset = ImageNet(imagenet_path)
    dataset.set_keep_ratio(args.keep_ratio)

    s_hits = torch.FloatTensor([0, 0, 0, 0, 0]).cuda() # top 1 2 5 10 20
    s_tot = 0

    results = {}


    for i, wnid in enumerate(test_wnids, 1):
        subset = dataset.get_subset(wnid)
        hits, tot = test_on_subset(subset, n, pred_vectors, n + i - 1,
                                       consider_trains=args.consider_trains)
        results[wnid] = (hits / tot).tolist()

        s_hits += hits
        s_tot += tot

        print('{}/{}, {}:'.format(i, len(test_wnids), wnid), end=' ')
        for i in range(len(hits)):
            print('{:.0f}%({:.2f}%)'.format(hits[i] / tot * 100, s_hits[i] / s_tot * 100), end=' ')
            print('x{}({})'.format(tot, s_tot))

    print('summary:', end=' ')
    for s_hit in s_hits:
        print('{:.2f}%'.format(s_hit / s_tot * 100), end=' ')
    print('total {}'.format(s_tot))

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))

