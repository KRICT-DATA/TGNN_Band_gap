import numpy
import util.crystal_conv as cc
import util.trainer as tr
import torch.nn
from torch.utils.data import DataLoader
from model.TGNN import TGNN


train_data, val_data, test_data = cc.load_dataset('../data/crystal/mp-ids-46744', target_idx=1, ref_idx=1, radius=4)
data = train_data + val_data + test_data
mp_ids = numpy.array([c.mp_id for c in data]).reshape(-1, 1)
model = TGNN(cc.num_atom_feats, cc.num_bond_feats, 1).cuda()
model.load_state_dict(torch.load('model.pt'))
model.eval()

criterion = torch.nn.L1Loss()
data_loader = DataLoader(data, batch_size=2, collate_fn=tr.collate)
train_loss, train_rmse, idxs, targets, preds = tr.test(model, data_loader, criterion)

file = open('pred_results_mp.csv', 'a')
numpy.savetxt(file, numpy.hstack([mp_ids, targets, preds]), delimiter=',', fmt='%s,%s,%s')
file.close()
