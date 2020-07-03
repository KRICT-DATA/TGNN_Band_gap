import numpy
import util.crystal_conv as cc
import util.trainer as tr
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.TGNN import TGNN


num_epochs = 300
batch_size = 32


cc.load_mat_atom_feats()
list_test_mae = list()
list_test_rmse = list()

for n in range(0, 10):
    train_data, val_data, test_data = cc.load_dataset('../data/crystal/c2db', target_idx=2, ref_idx=1, radius=5, train_ratio=0.5, val_ratio=0.3)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=tr.collate)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=tr.collate)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=tr.collate)

    model = TGNN(cc.num_atom_feats, cc.num_bond_feats, 1).cuda()
    # optimizer = AdaBound(model.parameters(), lr=1e-6, weight_decay=1e-8)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
    criterion = torch.nn.L1Loss()
    best_val_loss = 1e+8

    for epoch in range(0, num_epochs):
        train_loss = tr.train(model, optimizer, train_data_loader, criterion)
        val_loss, val_rmse, _, _, _ = tr.test(model, val_data_loader, criterion)
        test_loss, test_rmse, idxs, targets, preds = tr.test(model, test_data_loader, criterion)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}\tVal loss: {:.4f} ({:.4f})\tTest loss: {:.4f} ({:.4f})'
              .format(epoch + 1, num_epochs, train_loss, val_loss, val_rmse, test_loss, test_rmse))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_test_rmse = test_rmse
            best_idxs = idxs
            best_targets = targets
            best_preds = preds

    list_test_mae.append(best_test_loss)
    list_test_rmse.append(best_test_rmse)
    print(best_test_loss, best_test_rmse)

print(numpy.average(list_test_mae), numpy.average(list_test_rmse))
