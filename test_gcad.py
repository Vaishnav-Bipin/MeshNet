import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from config import get_test_config
from data import GrabCad33
from models import MeshNet
from utils.retrival import append_feature, calculate_map
from acc_measures import *


cfg = get_test_config(config_file='config/test_config_gcad33.yaml')
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']


data_set = GrabCad33(cfg=cfg['dataset'], part='test')
data_loader = data.DataLoader(data_set, batch_size=cfg['batch_size'], num_workers=4, shuffle=True, pin_memory=False)

acc_mat = np.zeros((33, 33))

def test_model(model):

    correct_num = 0
    ft_all, lbl_all = None, None

    with torch.no_grad():
        for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader):
            centers = centers.cuda()
            corners = corners.cuda()
            normals = normals.cuda()
            neighbor_index = neighbor_index.cuda()
            targets = targets.cuda()

            outputs, feas = model(centers, corners, normals, neighbor_index)
            _, preds = torch.max(outputs, 1)

            correct_num += (preds == targets).float().sum()

            for i in range(targets.shape[0]):
                pred = preds[i].item()
                true = targets[i].item()
                acc_mat[pred, true] += 1


            if cfg['retrieval_on']:
                ft_all = append_feature(ft_all, feas.detach().cpu())
                lbl_all = append_feature(lbl_all, targets.detach().cpu(), flaten=True)

    print('Accuracy: {:.4f}'.format(float(correct_num) / len(data_set)))
    if cfg['retrieval_on']:
        print('mAP: {:.4f}'.format(calculate_map(ft_all, lbl_all)))
    
    vers = cfg['vers']
    np.savetxt(f'acc_mat_{vers}.out', acc_mat)
    acc_avgs = np.array([precision_macavg(acc_mat), recall_macavg(acc_mat), f1_macavg(acc_mat), accuracy_avg(acc_mat)])
    np.savetxt(f'acc_avgs_{vers}.out', acc_avgs)



if __name__ == '__main__':

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(cfg['load_model']))
    model.eval()

    test_model(model)
