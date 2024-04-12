
#Importing Libraries

#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

#PyTorch_Geometric Utilities
from torch_geometric.nn import GENConv, global_mean_pool, SAGPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


#Starting Graph Convolutional Network with GENConv layers

def models_map(input_dim, output_dim, edge_dim, device): 
    dict = {'GCN':GCN(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'Pool':Pool(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device)}
    return dict

def debug_p(string, debug):
    if debug:
        print(string)
        return
    return

class Pool(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.2):
        super().__init__()
        self.conv1 = GENConv(in_dim, 256, edge_dim=edge_dim)
        self.pool1 = SAGPooling(256, ratio=0.5)

        self.conv2 = GENConv(256, 128, edge_dim=edge_dim)
        self.pool2 = SAGPooling(128, ratio=0.5)

        self.conv3 = GENConv(128, 64, edge_dim=edge_dim)
        self.pool3 = SAGPooling(64, ratio=0.5)

        self.conv4 = GENConv(64, 64, edge_dim=edge_dim)
        self.pool4 = SAGPooling(64, ratio=0.8)

        self.conv5 = GENConv(64, 64, edge_dim=edge_dim)
        self.pool5 = SAGPooling(64, ratio=0.8)

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, out_dim)

    

    def forward(self, x, edge_index, edge_attr, batch, debug=False):
        
        debug_p(f'Start {x.shape}', debug)
        x = F.relu(self.conv1(x, edge_index))
        debug_p(f'Conv1 {x.shape}', debug)

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        debug_p(f'Pool1 {x.shape}', debug)

        x = F.relu(self.conv2(x, edge_index))
        debug_p(f'Conv2 {x.shape}', debug)

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        debug_p(f'Pool2 {x.shape}', debug)

        x = F.relu(self.conv3(x, edge_index))
        debug_p(f'Conv3 {x.shape}', debug)

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        debug_p(f'Pool3 {x.shape}', debug)

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        debug_p(f'x1 {x1.shape}', debug)

        x = F.relu(self.conv4(x, edge_index))
        debug_p(f'Conv4 {x.shape}', debug)

        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        debug_p(f'Pool4 {x.shape}', debug)

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        debug_p(f'x2 {x2.shape}', debug)

        x = F.relu(self.conv5(x, edge_index))
        debug_p(f'Conv5 {x.shape}', debug)

        x, edge_index, _, batch, _, _ = self.pool5(x, edge_index, None, batch)
        debug_p(f'Pool5 {x.shape}', debug)

        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        debug_p(f'x3 {x3.shape}', debug)

        x = x1 + x2 + x3
        debug_p(f'x1+x2+x3 {x.shape}', debug)

        x = F.relu(self.lin1(x))
        debug_p(f'Lin1 {x.shape}', debug)

        x = F.dropout(x, p=0.5, training=self.training)
        debug_p(f'Dropout {x.shape}', debug)

        x = F.relu(self.lin2(x))
        debug_p(f'Lin2 {x.shape}', debug)

        x = F.log_softmax(self.lin3(x), dim=-1)
        debug_p(f'Lin3 {x.shape}\n', debug)

        return x

class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GENConv(in_dim, 64, edge_dim=edge_dim)
        self.conv2 = GENConv(64, 64, edge_dim=edge_dim)
        self.conv3 = GENConv(64, 128, edge_dim=edge_dim)
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, out_dim)

    

    def forward(self, x, edge_index, edge_attr, batch):
        
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        x = x.relu()
        
        # Perform global mean pooling
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=self.dropout)
        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.log_softmax(x, dim=-1)
        return x
