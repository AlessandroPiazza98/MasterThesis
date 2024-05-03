
#Importing Libraries

#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

#PyTorch_Geometric Utilities
import torch_geometric.utils
from torch_geometric.nn import GENConv, global_mean_pool, global_add_pool, global_max_pool, SAGPooling, TopKPooling, GCNConv
from torch_geometric.data import Data
import torch_geometric.transforms as T


#Starting Graph Convolutional Network with GENConv layers

def model_feat_map(input_dim, output_dim, edge_dim, device, hidden, dropout): 
    dict = {'GCN':GCN(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'Pool':Pool(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'GCN5':GCN5(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'SAGPool':SAGPool(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'GCN2':GCN2(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim, hidden=hidden, dropout=dropout).to(device),
            'GCNSAG':GCN2(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim, hidden=hidden, dropout=dropout).to(device)}
    return dict

def debug_p(string, debug):
    if debug:
        print(string)
        return
    return

class SAGPool(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GENConv(in_dim, 64, edge_dim=edge_dim)
        self.pool1 = SAGPooling(64, ratio=0.2)
        self.conv2 = GENConv(64, 64, edge_dim=edge_dim)
        self.pool2 = SAGPooling(64, ratio=0.2)
        self.conv3 = GENConv(64, 128, edge_dim=edge_dim)
        self.pool3 = SAGPooling(128, ratio=0.2)
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, out_dim)

    

    def forward(self, data, batch, debug=False):
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index

        debug_p(f'Start {x.shape}', debug)
        if torch.isnan(x).any():
              raise ValueError("Start")
        
        x = self.conv1(x, edge_index, edge_attr)
        debug_p(f'Conv1 {x.shape}', debug)
        if torch.isnan(x).any():
              raise ValueError("Conv1")

        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        debug_p(f'Pool1 {x.shape}', debug)
        if torch.isnan(x).any():
              raise ValueError("Pool1")

        x = x.relu()

        x = self.conv2(x, edge_index, edge_attr)
        debug_p(f'Conv2 {x.shape}', debug)
        if torch.isnan(x).any():
              raise ValueError("Conv2")

        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        debug_p(f'Pool2 {x.shape}', debug)
        if torch.isnan(x).any():
              raise ValueError("Pool2")

        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        debug_p(f'Conv3 {x.shape}', debug)
        if torch.isnan(x).any():
              raise ValueError("Conv3")

        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        debug_p(f'Pool3 {x.shape}', debug)
        if torch.isnan(x).any():
              raise ValueError("Pool3")
        x = x.relu()
        # Perform global mean pooling
        x = global_mean_pool(x, batch)
        if torch.isnan(x).any():
              raise ValueError("GMPool")
        debug_p(f'GMPool {x.shape}', debug)
       
        x = F.dropout(x, p=self.dropout)
        if torch.isnan(x).any():
              raise ValueError("Dropout")
        x = self.dense1(x)
        debug_p(f'Lin1 {x.shape}', debug)
        if torch.isnan(x).any():
              raise ValueError("Lin1")

        x = self.dense2(x)
        debug_p(f'Lin2 {x.shape}', debug)
        if torch.isnan(x).any():
              raise ValueError("Lin2")
        x = torch.log_softmax(x, dim=-1)
        return x

class Pool(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GENConv(in_dim, 64, edge_dim=edge_dim)
        self.pool1 = TopKPooling(64, ratio=0.2)
        self.conv2 = GENConv(64, 64, edge_dim=edge_dim)
        self.pool2 = TopKPooling(64, ratio=0.2)
        self.conv3 = GENConv(64, 128, edge_dim=edge_dim)
        self.pool3 = TopKPooling(128, ratio=0.2)
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, out_dim)

    

    def forward(self, data, batch, debug=False):
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index

        debug_p(f'Start {x.shape}', debug)
        x = self.conv1(x, edge_index, edge_attr)
        debug_p(f'Conv1 {x.shape}', debug)

        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        debug_p(f'Pool1 {x.shape}', debug)

        x = x.relu()

        x = self.conv2(x, edge_index, edge_attr)
        debug_p(f'Conv2 {x.shape}', debug)

        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        debug_p(f'Pool2 {x.shape}', debug)

        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        debug_p(f'Conv3 {x.shape}', debug)

        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        debug_p(f'Pool3 {x.shape}', debug)
        x = x.relu()
        # Perform global mean pooling
        x = global_mean_pool(x, batch)
        debug_p(f'GMPool {x.shape}', debug)
       
        x = F.dropout(x, p=self.dropout)
        x = self.dense1(x)
        debug_p(f'Lin1 {x.shape}', debug)

        x = self.dense2(x)
        debug_p(f'Lin2 {x.shape}', debug)
        x = torch.log_softmax(x, dim=-1)
        return x

class OldPool2(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.2):
        super().__init__()
        self.conv1 = GENConv(in_dim, 256, edge_dim=edge_dim)
        self.pool1 = SAGPooling(256, ratio=0.5)
        self.conv2 = GENConv(256, 128, edge_dim=edge_dim)
        self.pool2 = SAGPooling(128, ratio=0.5)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, out_dim)

    

    def forward(self, x, edge_index, edge_attr, batch, debug=False):
        
        debug_p(f'Start {x.shape}', debug)
        x = self.conv1(x, edge_index)
        debug_p(f'Conv1 {x.shape}', debug)

        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        debug_p(f'Pool1 {x.shape}', debug)

        x = self.conv2(x, edge_index)
        debug_p(f'Conv2 {x.shape}', debug)

        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        debug_p(f'Pool2 {x.shape}', debug)

        x = global_mean_pool(x, batch)
        debug_p(f'GMPool {x.shape}', debug)

        x = F.relu(self.lin1(x))
        debug_p(f'Lin1 {x.shape}', debug)

        x = F.log_softmax(self.lin2(x), -1)
        debug_p(f'Lin2 {x.shape}', debug)

        return x

class OldPool(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GENConv(in_dim, 512, edge_dim=edge_dim)
        self.pool1 = SAGPooling(512, ratio=0.5)

        self.conv2 = GENConv(512, 256, edge_dim=edge_dim)
        self.pool2 = SAGPooling(256, ratio=0.5)

        self.conv3 = GENConv(256, 128, edge_dim=edge_dim)
        self.pool3 = SAGPooling(128, ratio=0.5)

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

        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        debug_p(f'Pool1 {x.shape}', debug)

        x = F.relu(self.conv2(x, edge_index))
        debug_p(f'Conv2 {x.shape}', debug)

        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        debug_p(f'Pool2 {x.shape}', debug)

        x = F.relu(self.conv3(x, edge_index))
        debug_p(f'Conv3 {x.shape}', debug)

        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        debug_p(f'Pool3 {x.shape}', debug)

        '''
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

        '''

        x = global_mean_pool(x, batch)
        debug_p(f'GMPool {x.shape}', debug)

        x = F.relu(self.lin1(x))
        debug_p(f'Lin1 {x.shape}', debug)

        x = F.dropout(x, p=self.dropout)
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

    

    def forward(self, data, batch, debug=False):
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index

        debug_p(f'Start {x.shape}', debug)
        x = self.conv1(x, edge_index, edge_attr)
        debug_p(f'Conv1 {x.shape}', debug)

        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        debug_p(f'Conv2 {x.shape}', debug)

        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        debug_p(f'Conv3 {x.shape}', debug)

        x = x.relu()
        # Perform global mean pooling
        x = global_mean_pool(x, batch)
        debug_p(f'GMPool {x.shape}', debug)
       
        x = F.dropout(x, p=self.dropout)
        x = self.dense1(x)
        debug_p(f'Lin1 {x.shape}', debug)

        x = self.dense2(x)
        debug_p(f'Lin2 {x.shape}', debug)
        x = torch.log_softmax(x, dim=-1)
        return x

class GCN2(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, hidden, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GENConv(in_dim, hidden, edge_dim=edge_dim)
        self.norm1 = nn.BatchNorm1d(hidden)
        self.conv2 = GENConv(hidden, hidden*2, edge_dim=edge_dim)
        self.norm2 = nn.BatchNorm1d(hidden*2)
        self.conv3 = GENConv(hidden*2, hidden*4, edge_dim=edge_dim)
        self.norm3 = nn.BatchNorm1d(hidden*4)
        self.dense1 = nn.Linear(hidden*4, hidden*2)
        self.dense2 = nn.Linear(hidden*2, hidden)
        self.dense3 = nn.Linear(hidden, out_dim)

    

    def forward(self, data, batch, debug=False):
        
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index

        debug_p(f'Start {x.shape}', debug)
        x = self.conv1(x, edge_index, edge_attr)
        debug_p(f'Conv1 {x.shape}', debug)
        x = self.norm1(x)

        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        debug_p(f'Conv2 {x.shape}', debug)
        x = self.norm2(x)

        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        debug_p(f'Conv3 {x.shape}', debug)
        x = self.norm3(x)

        x = x.relu()

        x = global_mean_pool(x, batch)
        debug_p(f'GMPool {x.shape}', debug)
       
        x = F.dropout(x, p=self.dropout)
        return x

class GCNSAG(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, hidden, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GENConv(in_dim, hidden, edge_dim=edge_dim)
        self.norm1 = nn.BatchNorm1d(hidden)
        self.conv2 = GENConv(hidden, hidden*2, edge_dim=edge_dim)
        self.norm2 = nn.BatchNorm1d(hidden*2)
        self.conv3 = GENConv(hidden*2, hidden*4, edge_dim=edge_dim)
        self.norm3 = nn.BatchNorm1d(hidden*4)
        self.pool1 = SAGPooling(hidden*4, ratio=0.5)

    

    def forward(self, data, batch, debug=False):
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index

        debug_p(f'Start {x.shape}', debug)
        x = self.conv1(x, edge_index, edge_attr)
        debug_p(f'Conv1 {x.shape}', debug)
        x = self.norm1(x)

        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        debug_p(f'Conv2 {x.shape}', debug)
        x = self.norm2(x)

        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        debug_p(f'Conv3 {x.shape}', debug)
        x = self.norm3(x)

        x = x.relu()
        
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        debug_p(f'Pool1 {x.shape}', debug)

        x = global_mean_pool(x, batch)
        debug_p(f'GMPool {x.shape}', debug)
       
        x = F.dropout(x, p=self.dropout)
        return x

class GCN5(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GENConv(in_dim, 64, edge_dim=edge_dim)
        self.conv2 = GENConv(64, 64, edge_dim=edge_dim)
        self.conv3 = GENConv(64, 128, edge_dim=edge_dim)
        self.conv4 = GENConv(128, 256, edge_dim=edge_dim)
        self.conv5 = GENConv(256, 512, edge_dim=edge_dim)
        self.dense1 = nn.Linear(512, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, out_dim)

    def forward(self, data, batch, debug=False):
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index

        debug_p(f'Start {x.shape}', debug)
        x = self.conv1(x, edge_index, edge_attr)
        debug_p(f'Conv1 {x.shape}', debug)

        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        debug_p(f'Conv2 {x.shape}', debug)

        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        debug_p(f'Conv3 {x.shape}', debug)

        x = x.relu()
        x = self.conv4(x, edge_index, edge_attr)
        debug_p(f'Conv4 {x.shape}', debug)

        x = x.relu()
        x = self.conv5(x, edge_index, edge_attr)
        debug_p(f'Conv5 {x.shape}', debug)

        x = x.relu()
        # Perform global mean pooling
        x = global_mean_pool(x, batch)
        debug_p(f'GMPool {x.shape}', debug)
        
        x = F.dropout(x, p=self.dropout)
        x = self.dense1(x)
        debug_p(f'Lin1 {x.shape}', debug)

        x = self.dense2(x)
        debug_p(f'Lin2 {x.shape}', debug)
        x = self.dense3(x)

        debug_p(f'Lin3 {x.shape}', debug)
        x = torch.log_softmax(x, dim=-1)
        return x

class Classifier(torch.nn.Module):
    def __init__(self, out_dim, hidden, dropout):
        super().__init__()
        self.dropout = dropout
        self.dense1 = nn.Linear(hidden*4, hidden*2)
        self.dense2 = nn.Linear(hidden*2, hidden)
        self.dense3 = nn.Linear(hidden, out_dim)
    
    def forward(self, x, debug=False):
        x = x

        debug_p(f'Start {x.shape}', debug)
        x = self.dense1(x)
        debug_p(f'Lin1 {x.shape}', debug)
        x = F.dropout(x, p=self.dropout)

        x = self.dense2(x)
        debug_p(f'Lin2 {x.shape}', debug)
        x = F.dropout(x, p=self.dropout)

        x = self.dense3(x)
        debug_p(f'Lin3 {x.shape}', debug)

        x = torch.log_softmax(x, dim=-1)
        return x    
