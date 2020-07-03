import numpy
import torch
import torch.nn as nn

rmse = nn.MSELoss()


def train(model, optimizer, data_loader, criterion):
    model.train()
    train_loss = 0

    for batch in data_loader:
        pred = model(batch[0], batch[1], batch[2], batch[3], batch[5])
        loss = criterion(batch[4], pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()

    return train_loss / len(data_loader)


def test(model, data_loader, criterion):
    model.eval()
    test_loss = 0
    test_rmse = 0
    list_idxs = list()
    list_targets = list()
    list_preds = list()

    with torch.no_grad():
        for batch in data_loader:
            pred = model(batch[0], batch[1], batch[2], batch[3], batch[5])
            test_loss += criterion(batch[4], pred).detach().item()
            test_rmse = rmse(batch[4], pred).detach().item()

            list_idxs.append(batch[6].cpu().numpy().reshape(-1, 1))
            list_targets.append(batch[4].cpu().numpy().reshape(-1, 1))
            list_preds.append(pred.cpu().numpy())

    return test_loss / len(data_loader), numpy.sqrt(test_rmse), numpy.vstack(list_idxs), numpy.vstack(list_targets), numpy.vstack(list_preds)


def collate(batch):
    list_pairs = list()
    list_idx_pairs = list()
    list_triplets = list()
    list_idx_triplets = list()
    list_targets = list()
    list_ref_feats = list()
    list_idxs = list()

    for data in batch:
        list_pairs.append(data.pairs)
        list_idx_pairs.append(data.pairs.shape[0])
        list_triplets.append(data.triplets)
        list_idx_triplets.append(data.triplets.shape[0])
        list_targets.append(data.y)
        list_ref_feats.append(data.ref_feat)
        list_idxs.append(data.id)

    pairs = torch.cat(list_pairs, dim=0).cuda()
    idx_pairs = torch.tensor(list_idx_pairs).view(-1, 1).cuda()
    triplets = torch.cat(list_triplets, dim=0).cuda()
    idx_triplets = torch.tensor(list_idx_triplets).view(-1, 1).cuda()
    targets = torch.cat(list_targets, dim=0).cuda()
    ref_feats = torch.cat(list_ref_feats, dim=0).cuda()
    idxs = torch.tensor(list_idxs).view(-1, 1).cuda()

    return pairs, idx_pairs, triplets, idx_triplets, targets, ref_feats, idxs
