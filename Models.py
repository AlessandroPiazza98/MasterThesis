
#Importing Libraries

#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

#PyTorch_Geometric Utilities
import torch_geometric.utils
from torch_geometric.nn import GENConv, global_mean_pool, SAGPooling, TopKPooling, GCNConv
import torch_geometric.transforms as T


#Starting Graph Convolutional Network with GENConv layers

def models_map(input_dim, output_dim, edge_dim, device): 
    dict = {'GCN':GCN(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'Pool':Pool(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'GCN5':GCN5(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'SAGPool':SAGPool(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'ASAP':ASAP(input_dim, 256, output_dim, 5).to(device)}
    return dict

def debug_p(string, debug):
    if debug:
        print(string)
        return
    return

class GCN_ASAP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout=0.2, return_embeds=False):
        # Initialisation of self.convs, 
        # self.bns, and self.softmax.

        super(GCN_ASAP, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        # The log softmax layer
        self.softmax = None

        ## Note:
        ##  self.convs has num_layers GCNConv layers
        ##  self.bns has num_layers - 1 BatchNorm1d layers
        ##  For more information on GCNConv please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        ## For more information on Batch Norm1d please refer to the documentation: 
        ## https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

        # Construct all convs
        self.convs = torch.nn.ModuleList()

        # construct all bns
        self.bns = torch.nn.ModuleList()

        #For the first layer, we go from dimensions input -> hidden
        #For middle layers we go from dimensions hidden-> hidden
        #For the end layer we go from hidden-> output

        for l in range(num_layers):
          if l==0: #change input output dims accordingly
            self.convs.append(GCNConv(input_dim, hidden_dim))
          elif l == num_layers-1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
          else:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
          if l < num_layers-1: 
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.last_conv = GCNConv(hidden_dim, output_dim)
        self.log_soft = torch.nn.LogSoftmax()

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_weight):
        # This function that takes the feature tensor x and
        # edge_index tensor adj_t, and edge_weight and returns the output tensor.

        out = None

        for l in range(len(self.convs)-1):
          x = self.convs[l](x, adj_t, edge_weight)
          x = self.bns[l](x)
          x = F.relu(x)
          x = F.dropout(x, training=self.training)

        x = self.last_conv(x, adj_t, edge_weight)
        if self.return_embeds is True:
          out = x
        else: 
          out = self.log_soft(x)

        return out
    
class ASAP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(ASAP, self).__init__()

        # Node embedding model, initially input_dim=input_dim, output_dim = hidden_dim
        self.gnn_node = GCN_ASAP(input_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)
        # Note that the input_dim and output_dim are set to hidden_dim
        # for subsequent layers
        self.gnn_node_2 = GCN_ASAP(hidden_dim, hidden_dim,
        hidden_dim, num_layers, dropout, return_embeds=True)

        ##Set up pooling layer using ASAPool
        ## For more information please refere to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.ASAPooling
        self.asap = torch_geometric.nn.pool.ASAPooling(in_channels = 256, ratio = 0.5, dropout = 0.1, negative_slope = 0.2, add_self_loops = False)

        ## Initialize self.pool as a global mean pooling layer
        ## For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
        self.pool = global_mean_pool

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch, debug=False):
        # This function takes as input a 
        # mini-batch of graphs (torch_geometric.data.Batch) and 
        # returns the predicted graph property for each graph. 
        #
        # Since we are predicting graph level properties,
        # the output will be a tensor with dimension equaling
        # the number of graphs in the mini-batch

    
        # Extract important attributes of our mini-batch
        embed = x
        out = None

        ## Note:
        ## 1. We construct node embeddings using existing GCN model
        ## 2. We use the ASAPool module for soft clustering into a coarser graph representation. 
        ## For more information please refere to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.ASAPooling
        ## 3. After two cycles of this, we use the global pooling layer to aggregate features for each individual graph
        ## For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
        ## 4. We use a linear layer to predict each graph's property
        num_graphs = int(len(batch)/51)
        post_GCN_1 = self.gnn_node(embed, edge_index, edge_weight)
        post_pool_1 = self.asap(post_GCN_1, edge_index)
        post_GCN_2 = self.gnn_node_2(post_pool_1[0], post_pool_1[1], post_pool_1[2])
        post_pool_2 = self.asap(post_GCN_2, post_pool_1[1])
        ultimate_gcn = self.gnn_node_2(post_pool_2[0], post_pool_2[1], post_pool_2[2])

        glob_pool = self.pool(ultimate_gcn, post_pool_2[3], num_graphs)  
        out = self.linear(glob_pool)    

        return out

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

    

    def forward(self, x, edge_index, edge_attr, batch, debug=False):
        
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

    

    def forward(self, x, edge_index, edge_attr, batch, debug=False):
        
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

    

    def forward(self, x, edge_index, edge_attr, batch, debug=False):
        
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

    def forward(self, x, edge_index, edge_attr, batch, debug=False):

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