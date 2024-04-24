import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm, Linear
from torch_geometric.data import Data


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert targets to type long, as expected by nn.CrossEntropyLoss
        targets = targets.view(-1, 1).long()
        # Compute the log softmax over the inputs
        log_pt = F.log_softmax(inputs, dim=1)
        # Gather the log prob predictions for the correct classes
        log_pt = log_pt.gather(1, targets)
        log_pt = log_pt.view(-1)
        pt = log_pt.exp()
        
        # Compute the focal loss component
        focal_loss = -1 * (1 - pt) ** self.gamma * log_pt
        # Apply the alpha weighting
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = at * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        convs = [GCNConv(input_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + [GCNConv(hidden_dim, output_dim)]
        self.convs = torch.nn.ModuleList(convs)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)])
        self.dropout = dropout
        self.softmax = torch.nn.Softmax(dim=1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.convs):
          x = layer(x, edge_index)
          if i < len(self.convs)-1:
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        out = x
            
        return out


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.node_emb = Linear(input_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.classifier = Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x = self.node_emb(data.x)
        
        x = self.conv1(x, data.edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv2(x, data.edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv3(x, data.edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.classifier(x)
        return x

# testa att ändra

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_channels * num_heads)  # Adjust for multi-head output
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_channels * num_heads)  # Again, adjust for multi-head output
        self.conv3 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Apply the first GAT convolution layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)

        # Apply the second GAT convolution layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        # Apply the third GAT convolution layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        

        return x