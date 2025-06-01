import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, random_seed, bins, data):
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
        x = F.dropout(x, p=0.5, training=self.training)

        for layer in self.hidden_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.output_layer(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, random_seed, bins, data, num_heads):
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
            x = F.dropout(x, p=0.5, training=self.training)  # Adjust dropout probability as needed

        x = self.convs[-1](x, edge_index)
        return x

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, random_seed, bins, data, num_heads = None):
        super().__init__()
        torch.manual_seed(random_seed)
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(data.num_features, hidden_channels))
        for _ in range(num_layers):
            self.fcs(torch.nn.Linear(hidden_channels, hidden_channels))
        if bins != 'regression':
            self.fcs.append(torch.nn.Linear(hidden_channels, len(bins) +1 ))
        else:
            self.fcs.append(torch.nn.Linear(hidden_channels, 1))
    def forward(self, x, edge_index = None):
        for fc in self.fcs[:-1]:
            x = fc(x)
            x = F.elu(x)
            x = F.dropout(x, p = 0.5, training = self.training)
        x = self.fcs[-1](x)
        return x
