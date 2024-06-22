
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
import math

from loguru import logger
import sys

#Starting Graph Convolutional Network with GENConv layers
logger.remove()
logger.add(sys.stderr, level="INFO")


def model_feat_map(input_dim, output_dim, edge_dim, device, hidden, dropout): 
    dict = {'GCN':GCN(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim, hidden=hidden, dropout=dropout).to(device),
            'GCN_old':GCN_old(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'Pool':Pool(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'GCN5':GCN5(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'SAGPool':SAGPool(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim).to(device),
            'GCN2':GCN2(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim, hidden=hidden, dropout=dropout).to(device),
            'GCNSAG':GCN2(in_dim = input_dim, out_dim = output_dim, edge_dim=edge_dim, hidden=hidden, dropout=dropout).to(device),
            }
    return dict



def classifier_map(output_dim, device, hidden, dropout, ntoken, nhead, num_encoder_layers, dim_feedforward): 
    dict = {'Classifier':Classifier(output_dim, hidden, dropout).to(device),
            'ClassifierWin':ClassifierWin(output_dim, hidden, ntoken, dropout).to(device),
            'Transformer':TransformerAndClassifier(d_model= hidden, nhead= nhead, num_encoder_layers= num_encoder_layers, dim_feedforward= dim_feedforward, dropout= dropout,
                                                    max_len= 5000, classifier_layer_size= 64, batch_first=True, classes=output_dim, norm_first=False).to(device)} #TODO check for 4x in Transformer
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

class GCN_old(torch.nn.Module):
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

class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, hidden, dropout):
        super().__init__()
        hidden = int(hidden/4)
        self.dropout = dropout
        self.conv1 = GENConv(in_dim, hidden, edge_dim=edge_dim)
        self.norm1 = nn.BatchNorm1d(hidden)
        self.conv2 = GENConv(hidden, hidden*2, edge_dim=edge_dim)
        self.norm2 = nn.BatchNorm1d(hidden*2)
        self.conv3 = GENConv(hidden*2, hidden*4, edge_dim=edge_dim)
        self.norm3 = nn.BatchNorm1d(hidden*4)
    

    def forward(self, data, batch, debug=False):
        
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index

        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)

        x = nn.LeakyReLU()(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)

        x = nn.LeakyReLU()(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)

        x = nn.LeakyReLU()(x)

        x = global_mean_pool(x, batch)       
        x = F.dropout(x, p=self.dropout)
        return x

class GCN2(torch.nn.Module): #TODO remove
    def __init__(self, in_dim, out_dim, edge_dim, hidden, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GENConv(in_dim, hidden, edge_dim=edge_dim)
        self.norm1 = nn.BatchNorm1d(hidden)
        self.conv2 = GENConv(hidden, hidden*2, edge_dim=edge_dim)
        self.norm2 = nn.BatchNorm1d(hidden*2)
        self.conv3 = GENConv(hidden*2, hidden*4, edge_dim=edge_dim)
        self.norm3 = nn.BatchNorm1d(hidden*4)
    

    def forward(self, data, batch, debug=False):
        
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index

        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)

        x = nn.LeakyReLU()(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)

        x = nn.LeakyReLU()(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)

        x = nn.LeakyReLU()(x)

        x = global_mean_pool(x, batch)       
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

        x = self.conv1(x, edge_index, edge_attr)

        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)

        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)

        x = x.relu()
        x = self.conv4(x, edge_index, edge_attr)

        x = x.relu()
        x = self.conv5(x, edge_index, edge_attr)

        x = x.relu()
        # Perform global mean pooling
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=self.dropout)
        x = self.dense1(x)

        x = self.dense2(x)
        x = self.dense3(x)
        x = torch.log_softmax(x, dim=-1)
        return x 

class Classifier(torch.nn.Module): #TODO There is no RELU and correct the dimensions for hidden
    def __init__(self, out_dim, hidden, dropout):
        super().__init__()
        hidden = int(hidden/4)
        self.dropout = dropout
        self.dense1 = nn.Linear(hidden*4, hidden*2)
        self.dense2 = nn.Linear(hidden*2, hidden)
        self.dense3 = nn.Linear(hidden, out_dim)
    
    def forward(self, x, debug=False):
        x = x

        x = self.dense1(x)
        x = F.dropout(x, p=self.dropout)
        x = self.dense2(x)
        x = F.dropout(x, p=self.dropout)
        x = self.dense3(x)

        x = torch.log_softmax(x, dim=-1)
        return x    


class ClassifierWin(torch.nn.Module):
    def __init__(self, out_dim, hidden, tokens, dropout):
        super().__init__()
        hidden=int(hidden/4)
        self.dropout = dropout
        self.dense1 = nn.Linear(hidden*4*tokens, hidden*2*tokens)
        self.dense2 = nn.Linear(hidden*2*tokens, hidden*tokens)
        self.dense3 = nn.Linear(hidden*tokens, out_dim)
    
    def forward(self, x, debug=False):
        x = x

        x = self.dense1(x)
        x = F.dropout(x, p=self.dropout)
        x = self.dense2(x)
        x = F.dropout(x, p=self.dropout)
        x = self.dense3(x)

        x = torch.log_softmax(x, dim=-1)
        return x    

class TransformerAndClassifier(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float,
                 max_len: int, classifier_layer_size: int = 64, batch_first=True, classes=10, norm_first=False, *args,
                 **kwargs
                 ):
        super(TransformerAndClassifier, self).__init__()
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, dim_feedforward, dropout,
                                       max_len, batch_first, norm_first)
        self.classifier = ClassifierHead(d_model, classes, dropout=dropout, linear_size=classifier_layer_size)

    def forward(self, x):
        cls_token = self.transformer(x)
        x = self.classifier(cls_token)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        logger.debug(f'POSITIONAL ENCODING')
        logger.debug(f'x.shape: {x.shape}')
        logger.debug(f'pe.shape: {self.pe.shape}')
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):

    # FROM THE ATTENTION LAYER
    # key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
    #             to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
    #             Binary and float masks are supported.
    #             For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
    #             the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float,
                 max_len: int, batch_first=True, norm_first=False, *args, **kwargs
                 ):
        super().__init__(*args)
        self.norm_first = norm_first
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=batch_first, norm_first=self.norm_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.positional_encoder = PositionalEncoding(d_model, dropout=0.0, max_len=max_len)

        # self.fully_connected = nn.Linear(d_model, classes)
        # self.linear_classifier = nn.Linear(d_model, d_model)
        # self.pooling = nn.AvgPool1d(kernel_size=kwargs['num_words'], stride=1)
        self.softmax = nn.LogSoftmax(dim=1)

        self.class_token = torch.nn.Parameter(
                torch.randn(1, 1, d_model), requires_grad=True)
        torch.nn.init.normal_(self.class_token, std=0.02)
        self.input_layer = nn.Linear(d_model, d_model)
        self.input_layer.weight.data.normal_(mean=0., std=1.)
        self.input_layer.bias.data.zero_()

        self.batch_first = batch_first
        self.features = d_model

        # self.pos_embedding = torch.nn.Parameter(
        #         torch.randn(1, 2, n_channels)
        #         )

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None, ) -> torch.Tensor:
        # cls_token = torch.randn((x.shape[0], 1, x.shape[-1]), device=x.device)
        # tokens = torch.column_stack((cls_token, x))
        
        logger.debug(f'Transformer input shape: {src.shape}')
        logger.debug(
            f'Transformer input min: {torch.min(src)}, max: {torch.max(src)}, mean: {torch.mean(src)}, std: {torch.std(src)}')

        if torch.any(src.isnan()):
            logger.debug(f'Transf input has nan: {torch.any(src.isnan())}')
            raise NanError('NaN in Transformer input')

        # src = torch.nn.functional.batch_norm(src, torch.zeros(42, device=device), torch.ones(42, device=device))
        # logger.debug(f'Transformer input shape: {src.shape}')
        # logger.debug(f'Transformer input min: {torch.min(src)}, max: {torch.max(src)}, mean: {torch.mean(src)}, std: {torch.std(src)}')

        logger.debug(f'features {self.features}')
        # src = torch.movedim(src, (0, 1, 2), (0, 2, 1))
        # src = nn.BatchNorm1d(self.features, affine=True, device=device)(src)
        # src = torch.movedim(src, (0,1,2), (0,2,1))
        x = self.input_layer(src)

        if torch.any(x.isnan()):
            logger.error(f'Input Layers output x has nan: {torch.any(x.isnan())}')
            logger.error(
                f'input in transformer linear layer was: mean = {torch.mean(src)}, std = {torch.std(src)}, min = {torch.min(src)}, max = {torch.max(src)}')
            logger.error(f'weight in transformer linear layer has nans? {torch.any(self.input_layer.weight.isnan())}')
            raise NanError('NaN in Transformer input after Lin layer')

        # ADD CLS token
        x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        # ADD CLS Token in masks
        if src_key_padding_mask is not None:
            src_key_padding_mask = torch.cat(
                    [torch.logical_not(torch.ones((src_key_padding_mask.shape[0], 1), dtype=torch.bool, device=device)),
                     src_key_padding_mask], dim=1)
            logger.debug(f'transformer mask: {src_key_padding_mask.shape}')
        logger.debug(f'x expanded with CLS x.shape: {x.shape}')

        if torch.any(x.isnan()):
            logger.debug(f'Before Positional encoding x has nan: {torch.any(x.isnan())}')
            raise ValueError('NaN in Transformer Before Positional encoding')
        if self.batch_first:
            x = self.positional_encoder(x.permute(1, 0, 2))
            x = x.permute(1, 0, 2)
        else:
            x = self.positional_encoder(x)
        if torch.any(x.isnan()):
            logger.debug(f'Positional Encoded x has nan: {torch.any(x.isnan())}')
            raise ValueError('NaN in Transformer After Positional encoding')
        logger.debug(f'Positional encoding x.shape: {x.shape}')
        x = self.transformer_encoder(x, mask, src_key_padding_mask=src_key_padding_mask)

        # x = self.pooling(x.permute(0,2,1)).permute(0, 2, 1) # pool on the words dimensions -> it becomes one word
        logger.debug(f'Transformer encoder x.shape: {x.shape}')
        x = x[:, 0, :]  # extract CLS
        logger.debug(f'CLS extracted x.shape: {x.shape}')
        if torch.any(x.isnan()):
            raise ValueError('NaN in Transformer CLS token')
        # x = self.linear_classifier(x)
        # x = nn.ReLU()(x)
        # x = self.fully_connected(x)
        # x = torch.squeeze(x)

        return x

class ClassifierHead(nn.Module):
    def __init__(self, feature_input, out_classes, linear_size=64, hidden_size=256, dropout=0.1, linear=False):
        super().__init__()
        self.linear_classifier = linear
        self.linear_size = linear_size
        self.feature_input = feature_input
        self.output_size = out_classes

        self.linear = nn.Linear(feature_input, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, linear_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(linear_size, out_classes, bias=True)

        if self.linear_classifier:
            self.layers = nn.Sequential(nn.Linear(feature_input, out_classes, bias=True))
        else:
            self.layers = nn.Sequential(self.linear, self.dropout, self.activation, self.linear2, self.dropout,
                                        self.activation, self.output)

    def forward(self, x):
        logger.debug(f"input in classifier head {x.shape}")
        x = self.layers(x)
        logger.debug(f"output in classifier head {x.shape}")
        return x
    

# Define the Encoder (aligned with GCN2 model)
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden, edge_dim, out_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.conv1 = GENConv(in_dim, hidden, edge_dim=edge_dim)
        self.norm1 = nn.BatchNorm1d(hidden)
        self.conv2 = GENConv(hidden, hidden, edge_dim=edge_dim)
        self.norm2 = nn.BatchNorm1d(hidden)
        self.conv3 = GENConv(hidden, hidden, edge_dim=edge_dim)
        self.norm3 = nn.BatchNorm1d(hidden)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
        x = F.relu(x)

        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x

# Define the Decoder
class GraphDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, dropout):
        super(GraphDecoder, self).__init__()
        self.dropout = dropout
        self.lin = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GENConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GENConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GENConv(hidden_dim, output_dim, edge_dim=edge_dim)
        self.norm3 = nn.BatchNorm1d(output_dim)

    def forward(self, z, data):
        z = self.lin(z)
        z = F.relu(z)
        z = z[data.batch] #TODO
        x = self.conv1(z, data.edge_index, data.edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv3(x, data.edge_index, data.edge_attr)
        x = self.norm3(x)
        return x

# Define the Autoencoder
class GraphAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GraphAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        z = self.encoder(data)
        recon_x = self.decoder(z, data)
        return recon_x
