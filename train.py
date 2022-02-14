import argparse
import json
import random
import os.path as osp
import torch
import torch.nn.functional as F

from utils import ensure_path, set_gpu, l2_loss
from gcn import GCN_Dense

def save_checkpoint(name):
    torch.save(pred_obj, osp.join(save_path, name + '.pred'))
def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=1000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--save-epoch', type=int, default=200)
    parser.add_argument('--save-path', default='baseline')

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--no-pred', action='store_true')
    args = parser.parse_args()


    set_gpu(args.gpu)

    save_path = args.save_path
    ensure_path(save_path)

    #load graph
    graph = json.load(open('C:/Users/Lenovo/Desktop/part/graph/svkg.json', 'r'))
    svkg_wnids = graph['wnids'] + graph['visual'] + graph['aug']
    n = len(svkg_wnids)

    # X_s + X_v + X_aug
    svkg_vectors = torch.cat((torch.tensor(graph['vectors']),torch.tensor(graph['visual_vectors'])),0)
    svkg_vectors = torch.cat((svkg_vectors, torch.tensor(graph['aug_vector'])))
    svkg_vectors = F.normalize(svkg_vectors).cuda()

    # A_s + A_v + A_aug
    svkg_edges = graph['edges'] + graph['visual_ed'] + graph['aug_ed']
    svkg_edges = graph['edges'] + [(v, u) for (u, v) in graph['edges']]
    svkg_edges =  svkg_edges + [(u, u) for u in range(n)]
    n = len(svkg_wnids)

    #target: real visual feature
    fcfile = json.load(open('C:/Users/Lenovo/Desktop/part/visual/fc-weights.json', 'r'))
    train_wnids = [x[0] for x in fcfile]
    fc_vectors = [x[1] for x in fcfile]
    fc_vectors = torch.tensor(fc_vectors).cuda()
    fc_vectors = F.normalize(fc_vectors)

    hidden_layers = 'd2048,d'
    in_dim = svkg_vectors.shape[1]
    out_dim = fc_vectors.shape[1]

    print('{} nodes, {} edges'.format(n, len(svkg_edges)))
    print('word vectors:', svkg_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    # multimodal-gcn
    gcn = GCN_Dense(n,svkg_edges,in_dim,out_dim,hidden_layers).cuda()
    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    v_train, v_val = map(float, args.trainval.split(','))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    #semactic-visual transfer
    for epoch in range(1, args.max_epoch + 1):
        gcn.train()
        h = gcn(svkg_vectors)
        loss = mask_l2_loss(h, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gcn.eval()
        h = gcn(svkg_vectors)
        train_loss = mask_l2_loss(h, fc_vectors, tlist[:n_train]).item()
        if v_val > 0:
            val_loss = train_loss
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss
        print('epoch {}, train_loss={:.4f}'
              .format(epoch, train_loss))
        torch.cuda.empty_cache()

        trlog['train_loss'].append(train_loss)
        trlog['val_loss'].append(val_loss)
        trlog['min_loss'] = min_loss
        torch.save(trlog, osp.join(save_path, 'trlog'))

        #output: graph feature
        if (epoch % args.save_epoch == 0):
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': graph['wnids'],
                    'pred': h[:len(graph['wnids'])]
                }
        if epoch % args.save_epoch == 0:
            save_checkpoint('gcn13-{}'.format(epoch))

        pred_obj = None