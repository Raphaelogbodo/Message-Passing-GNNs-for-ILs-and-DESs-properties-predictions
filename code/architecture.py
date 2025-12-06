import requests
import os, sys
from pathlib import Path
import torch
import torch_geometric
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout, Sequential, ReLU, ModuleList
from torch_geometric.data import Data, DataLoader, InMemoryDataset, Dataset, Batch
from torch_geometric.nn import NNConv, GCNConv, MFConv, GATConv, CGConv, GraphConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, TopKPooling, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn import GraphNorm, LayerNorm
from torchmetrics import R2Score, MeanAbsoluteError, MeanAbsolutePercentageError
from permetrics.regression import RegressionMetric

'''Model Architectures'''

'''
1. Dynamic and Flexible NNConv Model
Embedding and Linear sizes are user-defined
'''
class NNConvModel(torch.nn.Module):
    def __init__(self,
                 input_channels=None,     
                 edge_channels=None,     
                 embedding_size=None,    
                 linear_size=None,       
                 add_params_num=None,    
                 dropout1=0.2,
                 dropout2=0.3,
                 aggr='mean',
                 pooling='mean',
                 norm_type='graph'):      # 'batch', 'layer', or 'graph'
        super().__init__()

        self.add_params_num = add_params_num
        self.dropout1 = Dropout(dropout1)
        self.dropout2 = Dropout(dropout2)
        self.pooling = pooling.lower()
        self.norm_type = norm_type.lower()

        # Optional attention pooling
        if self.pooling == 'attention':
            self.gate_nn = nn.Sequential(
                nn.Linear(embedding_size[-1], embedding_size[-1]),
                nn.ReLU(),
                nn.Linear(embedding_size[-1], 1)
            )
            self.att_pool = AttentionalAggregation(self.gate_nn)

        # GNN Layers
        self.conv_layers = ModuleList()
        self.norm_layers = ModuleList()  # store GraphNorms or LayerNorms

        in_channels = input_channels
        for out_channels in embedding_size:
            hidden_edge = max(32, out_channels // 4)
            nn_edge = Sequential(
                Linear(edge_channels, hidden_edge),
                ReLU(),
                Linear(hidden_edge, in_channels * out_channels)
            )
            conv = NNConv(in_channels, out_channels, nn_edge, aggr=aggr)
            self.conv_layers.append(conv)

            # choose norm type for graph-level layers
            if self.norm_type == 'graph':
                self.norm_layers.append(GraphNorm(out_channels))
            elif self.norm_type == 'layer':
                self.norm_layers.append(LayerNorm(out_channels))
            else:
                self.norm_layers.append(BatchNorm1d(out_channels))

            in_channels = out_channels  # next input = current output

        # Linear Layers
        if self.pooling == 'mean_max':
            in_dim = embedding_size[-1] * 2 + self.add_params_num
        else:
            in_dim = embedding_size[-1] + self.add_params_num

        self.linears = ModuleList()
        self.linear_norms = ModuleList()
        prev_dim = in_dim
        for out_dim in linear_size:
            self.linears.append(Linear(prev_dim, out_dim))
            if self.norm_type == 'batch':
                self.linear_norms.append(BatchNorm1d(out_dim))
            else:
                self.linear_norms.append(LayerNorm(out_dim))
            prev_dim = out_dim

        # Output layer
        self.out = Linear(linear_size[-1], 1)

    # Forward Pass
    def forward(self, x, edge_index, edge_attr, batch, cond=None, global_feats=None):
        # GNN Message Passing
        h = x
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            h = conv(h, edge_index, edge_attr)
            h = norm(h, batch) if isinstance(norm, GraphNorm) else norm(h)
            h = F.relu(h)
            h = self.dropout1(h)

        # Pooling
        if self.pooling == 'mean':
            pooled = global_mean_pool(h, batch)
        elif self.pooling == 'add':
            pooled = global_add_pool(h, batch)
        elif self.pooling == 'max':
            pooled = global_max_pool(h, batch)
        elif self.pooling == 'mean_max':
            pooled = torch.cat(
                [global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1
            )
        elif self.pooling == 'attention':
            pooled = self.att_pool(h, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Add external (conditional) inputs 
        if self.add_params_num != 0:
            if cond is not None:
                pooled = torch.cat([pooled, cond], dim=1)
            if global_feats is not None:
                pooled = torch.cat([pooled, global_feats], dim=1)

        # Feedforward (MLP)
        h = pooled
        for linear, norm in zip(self.linears, self.linear_norms):
            h = linear(h)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout2(h)

        out = self.out(h)
        return out, h


'''
2. Generalized Dynamic GCNConv Model
Embedding and Linear sizes are user-defined
'''

class GCNConvModel(torch.nn.Module):
    """
    Fully dynamic GCN-based GNN model.
    - Builds any number of GCNConv layers from embedding_size.
    - Builds any number of Linear layers from linear_size.
    - Supports edge_attr projection, dropout, batchnorm, and optional global features.
    """
    def __init__(self,
                 input_channels=None,     
                 edge_channels=None,       
                 embedding_size=None,  
                 linear_size=None,             
                 add_params_num=None,        
                 dropout1=0.2,
                 dropout2=0.3,
                pooling='mean'):
        super().__init__()

        self.add_params_num = add_params_num
        self.dropout1 = Dropout(dropout1)
        self.dropout2 = Dropout(dropout2)

        self.pooling = pooling.lower()

        # Optional attention pooling
        if self.pooling == 'attention':
            self.gate_nn = nn.Sequential(
                nn.Linear(embedding_size[-1], embedding_size[-1]),
                nn.ReLU(),
                nn.Linear(embedding_size[-1], 1)
            )
            self.att_pool = AttentionalAggregation(self.gate_nn)

        # Edge MLP 
        self.edge_mlp = Sequential(
            Linear(edge_channels, max(16, edge_channels * 2)),
            ReLU(),
            Linear(max(16, edge_channels * 2), 1)
        )
        
        #  GCN Layers 
        self.convs = ModuleList()
        self.norms = ModuleList()
        in_channels = input_channels
        for out_channels in embedding_size:
            self.convs.append(GCNConv(in_channels, out_channels))
            self.norms.append(GraphNorm(out_channels))
            in_channels = out_channels

        # Linear Layers 
        if self.pooling == 'mean_max':
            # After pooling, input size = 2 * last_embedding + add_params_num (if cond/global_feats)
            lin_in = embedding_size[-1] * 2 + (self.add_params_num if self.add_params_num else 0)
        else:
            lin_in = embedding_size[-1] + (self.add_params_num if self.add_params_num else 0)
    
        self.linears = ModuleList()
        self.norms_lin = ModuleList()
        prev_dim = lin_in
        for hdim in linear_size:
            self.linears.append(Linear(prev_dim, hdim))
            self.norms_lin.append(BatchNorm1d(hdim))
            prev_dim = hdim

        self.out = Linear(linear_size[-1], 1)

    def forward(self, x, edge_index, edge_attr, batch, cond=None, global_feats=None):
        """
        x: [num_nodes, node_feat_dim]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_channels]
        batch: [num_nodes]
        cond/global_feats: optional per-graph tensors
        """

        # Edge weights 
        if edge_attr is not None:
            edge_weight = self.edge_mlp(edge_attr).view(-1)
        else:
            edge_weight = None

        # GCN layers 
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_weight)
            h = norm(h, batch)
            h = F.relu(h)
            h = self.dropout1(h)

        # Unified pooling control
        if self.pooling == 'mean':
            pooled = global_mean_pool(h, batch)
        elif self.pooling == 'add':
            pooled = global_add_pool(h, batch)
        elif self.pooling == 'mean_max':
            pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        elif self.pooling == 'max':
            pooled = global_max_pool(h, batch)
        elif self.pooling == 'attention':
            pooled = self.att_pool(h, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Add conditional/global features 
        if cond is not None:
            pooled = torch.cat([pooled, cond], dim=1)
        if global_feats is not None:
            pooled = torch.cat([pooled, global_feats], dim=1)
            
        h = pooled
        #  Linear layers 
        for linear, norm in zip(self.linears, self.norms_lin):
            h = linear(h)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout2(h)

        out = self.out(h)
        return out, h

'''TRANSFER MODEL'''

class TransferGNN(nn.Module):
    def __init__(self, encoder, head_hidden=None, add_params_num=0, freeze_encoder=False):
        """
        encoder: pretrained encoder model (returns out, embedding)
        head_hidden: list of hidden sizes for the transfer head
        add_params_num: number of extra per-graph features (cond/global_feats)
        freeze_encoder: whether to freeze encoder weights
        """
        super().__init__()
        self.encoder = encoder
        self.add_params_num = add_params_num

        # Infer encoder output dimension automatically
        # Assume encoder.linears[-1].out_features = last hidden size before encoder.out
        if hasattr(self.encoder, 'linears') and len(self.encoder.linears) > 0:
            encoder_output_dim = self.encoder.linears[-1].out_features
        else:
            raise ValueError("Encoder must have linears attribute with at least one layer")

        in_dim = encoder_output_dim + (add_params_num if add_params_num else 0)

        # Build head dynamically
        layers = []
        prev = in_dim
        for h in head_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.3))
            prev = h
        layers.append(nn.Linear(prev, 1))  # final output
        self.head = nn.Sequential(*layers)

        # Freeze encoder if requested
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self._set_encoder_bn_eval()

    def _set_encoder_bn_eval(self):
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

    def forward(self, x, edge_index, edge_attr, batch, cond=None, global_feats=None):
        # Pass through encoder
        _, emb = self.encoder(x, edge_index, edge_attr, batch, cond=cond, global_feats=global_feats)

        # Concatenate extra per-graph features if present
        extra_features = []
        if cond is not None:
            extra_features.append(cond)
        if global_feats is not None:
            extra_features.append(global_feats)
        if extra_features:
            emb = torch.cat([emb] + extra_features, dim=1)

        # Pass through transfer head
        out = self.head(emb)
        return out, emb

