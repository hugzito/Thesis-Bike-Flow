import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def stratified_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=100):
    positive_mask = data.y > 0
    positive_indices = positive_mask.nonzero(as_tuple=False).squeeze()
    negative_indices = (~positive_mask).nonzero(as_tuple=False).squeeze()

    pos_train, pos_temp = train_test_split(positive_indices, train_size=train_ratio, random_state=random_seed)
    pos_val, pos_test = train_test_split(pos_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_seed)

    neg_train, neg_temp = train_test_split(negative_indices, train_size=train_ratio, random_state=random_seed)
    neg_val, neg_test = train_test_split(neg_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_seed)

    train_idx = torch.cat([pos_train, neg_train])
    val_idx = torch.cat([pos_val, neg_val])
    test_idx = torch.cat([pos_test, neg_test])

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data

def stratified_kfold_split(data, num_folds=5, random_seed=100):
    """Generates K stratified folds based on whether y > 0."""

    from sklearn.model_selection import StratifiedKFold

    positive_mask = data.y > 0
    labels = positive_mask.long().cpu().numpy()  # Labels: 1 for positive, 0 for negative
    indices = np.arange(data.num_nodes)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    folds = []

    for train_idx, val_idx in skf.split(indices, labels):
        fold = {}
        fold['train_mask'] = torch.zeros(data.num_nodes, dtype=torch.bool)
        fold['val_mask'] = torch.zeros(data.num_nodes, dtype=torch.bool)

        fold['train_mask'][train_idx] = True
        fold['val_mask'][val_idx] = True

        folds.append(fold)

    return folds

def train(model, data, optimizer, criterion, device, bins):
    model.train()
    optimizer.zero_grad()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.test_mask = data.test_mask.to(device)
    data.y = data.y.to(device)
    optimizer.zero_grad()  # Clear gradients.
    mask = data.train_mask.squeeze().to(device) & (data.y > 0).squeeze().to(device)
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    # Convert target to 1D tensor with dtype=torch.long
    if bins == 'regression':
        target = data.y[mask].squeeze()
        loss = criterion(out[mask].squeeze(), target)  # Ensure target is 1D and long
    else:
        target = torch.bucketize(data.y[mask], bins).squeeze()
        loss = criterion(out[mask], target.long())
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss.item()

def test(model, data, criterion, device, bins):
    model.eval()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.val_mask = data.val_mask.to(device)
    data.y = data.y.to(device)
    mask = data.val_mask.squeeze() & (data.y > 0).squeeze()
    out = model(data.x, data.edge_index)
    #target = torch.bucketize(data.y[mask], bins).squeeze()
    if bins == 'regression':
        target = data.y[mask].squeeze()
        loss = criterion(out[mask].squeeze(), target)
        accuracy = r2_score(target.detach().cpu().numpy(), out[mask].detach().cpu().numpy())
    else:
        target = torch.bucketize(data.y[mask], bins).squeeze()
        loss = criterion(out[mask], target.long())  # Ensure target is 1D and long
        correct_preds = out[mask].argmax(dim=1)
        correct = (correct_preds == target).sum()
        accuracy = correct.item() / mask.sum().item()
    return accuracy, out, loss.item()

def try_int(i):
    try:
        return int(i)
    except:
        return str(i)
