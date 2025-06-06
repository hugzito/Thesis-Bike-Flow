import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import wandb
from sklearn.metrics import r2_score
import random

def try_int(i):
    try:
        return int(i)
    except:
        return str(i)

import subprocess
import sys

#subprocess.check_call([sys.executable,  '-m', 'pip', 'install', '--force-reinstall', 'numpy'])


# --- Configurations ---
epochs = int(os.getenv("EPOCHS", 10))
learning_rate = float(os.getenv("LEARNING_RATE", 0.001))
hidden_c = int(os.getenv("HIDDEN_C", 16))
random_seed = int(os.getenv("RANDOM_SEED", 42))
bins = [try_int(i) for i in os.getenv("BINS", "1000 5000 10000").split()]
num_layers = int(os.getenv("NUM_LAYERS", 5))
nh = int(os.getenv("NUM_HEADS", 10))
use_gat = try_int(os.getenv("GAT", 0))
api_key = os.getenv("API_KEY", None)
graph_num = os.getenv("GRAPH_NUM", 2)
dropout_p = float(os.getenv("DROPOUT", 0.5))
model_name = os.getenv("MODEL_NAME", None)
prefix = os.getenv("PREFIX", "best_loss")

# --- Initialize WandB ---
if bins[0] == 'REGRESSION':
    bins = 'regression'
if use_gat in[0, 1]:
    use_gat = bool(use_gat)

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)

if bins != 'regression':
    bins = torch.tensor(bins, device=device)

# === Parameters ===
graph_model_path = f'data/graphs/{graph_num}/models/{model_name}_{prefix}.pt'
explanation_output_path = f'data/graphs/{graph_num}/explanations/{model_name}/graph_level_explanation.json'
os.makedirs(os.path.dirname(explanation_output_path), exist_ok=True)

# --- Load Graph Data ---
with open(f'data/graphs/{graph_num}/linegraph_tg.pkl', 'rb') as f:
    data = pickle.load(f)
explainer_epochs = data.x.shape[1] if model_name else 200  # Use data.x.shape[1] if model_name is provided

data.edge_index = data.edge_index.contiguous()
data.x = data.x.contiguous()
data.y = data.y.contiguous()
print(data.x.shape, data.edge_index.shape, data.y.shape, flush=True)

torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

# --- Data Split ---
def stratified_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
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

def stratified_kfold_split(data, num_folds=5):
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

data = stratified_split(data)

# --- Model Definitions ---
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        torch.manual_seed(random_seed)
        self.input_layer = GCNConv(data.num_features, hidden_channels, improved=True, cached=True)

        # Create intermediate hidden layers (optional)
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(GCNConv(hidden_channels, hidden_channels, improved=True, cached=True))
        if bins != 'regression':
            self.output_layer = GCNConv(hidden_channels, len(bins) + 1, cached=True)
        else:
            self.output_layer = GCNConv(hidden_channels, 1, cached=True)


    def forward(self, x, edge_index):
        x = self.input_layer(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=dropout_p, training=self.training)

        for layer in self.hidden_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=dropout_p, training=self.training)

        x = self.output_layer(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self,hidden_channels, num_layers, num_heads):
        super().__init__()
        torch.manual_seed(random_seed)  # Replace with your desired seed

        self.convs = torch.nn.ModuleList()

        # Input layer
        self.convs.append(GATConv(data.num_features, hidden_channels, heads=num_heads, concat=True))

        # Hidden layers
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True))

        # Output layer
        if bins != 'regression':
            self.convs.append(GATConv(hidden_channels * num_heads, len(bins) + 1, heads=1, concat=False))
        else:
            self.convs.append(GATConv(hidden_channels * num_heads, 1, heads=1, concat=False))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=dropout_p, training=self.training)  # Adjust dropout probability as needed

        x = self.convs[-1](x, edge_index)
        return x

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, num_heads=None):
        super().__init__()
        torch.manual_seed(random_seed)
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(data.num_features, hidden_channels))
        for _ in range(num_layers):
            self.fcs.append(torch.nn.Linear(hidden_channels, hidden_channels))
        if bins != 'regression':
            self.fcs.append(torch.nn.Linear(hidden_channels, len(bins) + 1))
        else:
            self.fcs.append(torch.nn.Linear(hidden_channels, 1))

    def forward(self, x, edge_index=None):
        for fc in self.fcs[:-1]:
            x = fc(x)
            x = F.elu(x)
            x = F.dropout(x, p=dropout_p, training=self.training)
        x = self.fcs[-1](x)
        return x

# --- Model Instantiation ---

model = torch.load(f"data/graphs/{graph_num}/models/{model_name}.pt") if model_name else None
model = model.to(device)
model.load_state_dict(torch.load(f'data/graphs/{graph_num}/models/{model_name}_{prefix}.pt', map_location=device))


print(model, flush=True)

# Move data to device
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
criterion = torch.nn.CrossEntropyLoss()
if bins == 'regression':
    criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=epochs//100)



from torch_geometric.explain import GNNExplainer, Explainer

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=1000),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type=None,
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)


mask = data.val_mask.squeeze() & (data.y > 0).squeeze()
node_idx = torch.nonzero(mask, as_tuple=True)[0]

data_x = data.x.to(device)
edge_index = data.edge_index.to(device)
data_y = data.y.to(device)


print("🔍 Running graph-level explanation...", flush=True)

try:
    explanation = explainer(data.x.to(device), data.edge_index.to(device))
finally:
    # even if explainer fails, you might want to handle that separately
    pass
import os
import json
import torch

def save_explanation_as_json_val_nodes(explanation, data, output_path):
    """
    Saves an Explanation object as a JSON file, but only includes nodes in the validation mask.
    
    Parameters:
    - explanation: PyTorch Geometric Explanation object.
    - data: PyTorch Geometric Data object (must have val_mask).
    - output_path: Path to save the JSON file.
    """
    # Convert explanation to dict
    explanation_dict = explanation.to_dict()
    
    # Get validation mask indices
    val_mask = data.val_mask
    val_indices = torch.nonzero(val_mask, as_tuple=True)[0]
    
    def convert(item):
        if isinstance(item, torch.Tensor):
            # If it's a node-level tensor (1D or 2D with first dim == num_nodes), filter
            if item.size(0) == data.num_nodes:
                item = item[val_indices]
            return item.tolist()
        elif isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert(v) for v in item]
        else:
            return item

    explanation_serializable = convert(explanation_dict)
    
    # Write to JSON file
    temp_path = output_path + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(explanation_serializable, f, indent=2)
    os.replace(temp_path, output_path)
    
    print(f"✅ Explanation saved safely to {output_path}", flush=True)

save_explanation_as_json_val_nodes(explanation, data, explanation_output_path)

import torch
import numpy as np
import pandas as pd

def compute_per_bin_gradients_val_nodes(model, data, device='cpu', feature_names=None):
    """
    Computes the per-bin gradients of the model's output with respect to the input features
    for all validation nodes where data.y > 0.

    Parameters:
    - model: Trained PyTorch model.
    - data: PyTorch Geometric Data object with x, edge_index, val_mask, y
    - device: The device to run on ('cpu' or 'cuda').
    - feature_names: List of feature names (optional).

    Returns:
    - pd.DataFrame: A dataframe where each row is (node_index, bin_index) and columns are features.
    """
    model = model.to(device)
    model.eval()

    # Ensure data.x requires gradients
    data_x = data.x.to(device)
    data_x.requires_grad_(True)
    edge_index = data.edge_index.to(device)

    # Filter validation nodes with y > 0
    mask = (data.val_mask & (data.y > 0)).squeeze()
    node_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()

    # Determine the number of bins (classes)
    with torch.no_grad():
        out = model(data_x, edge_index)  # try once to get shape
        num_bins = out.shape[1]

    records = []

    for node_idx in node_indices:
        for bin_idx in range(num_bins):
            model.zero_grad()
            try:
                out = model(data_x, edge_index)
            except ValueError as e:
                if "too many values to unpack" in str(e):
                    out = model((data_x, data_x), edge_index)
                else:
                    raise e

            out[node_idx, bin_idx].backward(retain_graph=True)

            grad = data_x.grad[node_idx].detach().cpu().numpy().flatten()
            record = {
                'node_index': node_idx,
                'bin_index': bin_idx
            }

            if feature_names is None:
                feature_cols = {f'feature_{i}': grad[i] for i in range(len(grad))}
            else:
                feature_cols = {feature_names[i]: grad[i] for i in range(len(grad))}

            record.update(feature_cols)
            records.append(record)

            # Zero gradients for next bin
            data_x.grad.zero_()

    df = pd.DataFrame.from_records(records)
    return df

# mask = data.val_mask.squeeze() & (data.y > 0).squeeze()
gradient_df = compute_per_bin_gradients_val_nodes(
    model,
    data,
    feature_names=data.feat_names  # replace with your feature names if available
)

print(gradient_df.head())
# Save the gradient DataFrame to a CSV file
gradient_output_path = f'data/graphs/{graph_num}/explanations/{model_name}/per_bin_gradients_all_nodes.csv'
gradient_df.to_csv(gradient_output_path, index=False)