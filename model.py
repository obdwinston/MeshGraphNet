import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

NODE_NORMAL, NODE_OUTFLOW, NODE_TYPES = 0, 5, 9


def mlp(in_dim, out_dim, hidden=128, layer_norm=True):
    layers = [
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    ]
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class Normaliser(nn.Module):
    def __init__(self, size, max_n=1e6, eps=1e-8):
        super().__init__()
        self.max_n = max_n
        self.eps = eps
        self.register_buffer("n", torch.zeros(1))
        self.register_buffer("sum", torch.zeros(size))
        self.register_buffer("sum_sq", torch.zeros(size))

    def mean(self):
        return self.sum / self.n.clamp(min=1.0)

    def std(self):
        var = self.sum_sq / self.n.clamp(min=1.0) - self.mean() ** 2
        return torch.sqrt(var.clamp(min=self.eps))

    def forward(self, x, accumulate=True):
        if accumulate and self.training and self.n.item() < self.max_n:
            self.n += x.shape[0]
            self.sum += x.sum(0)
            self.sum_sq += (x * x).sum(0)
        return (x - self.mean()) / self.std()

    def inverse(self, x):
        return x * self.std() + self.mean()


class ProcessorLayer(MessagePassing):
    def __init__(self, hidden):
        super().__init__(aggr=None)
        self.edge_mlp = mlp(3 * hidden, hidden, hidden)
        self.node_mlp = mlp(2 * hidden, hidden, hidden)

    def message(self, x_i, x_j, edge_attr):
        return edge_attr + self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=1))

    def aggregate(self, updated_edges, edge_index):
        aggregated = scatter(updated_edges, edge_index[0], dim=0, reduce="sum")
        return aggregated, updated_edges

    def forward(self, x, edge_index, edge_attr):
        aggregated, updated_edges = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        updated_nodes = x + self.node_mlp(torch.cat([x, aggregated], dim=1))
        return updated_nodes, updated_edges


class MeshGraphNet(nn.Module):
    def __init__(self, node_in=11, edge_in=3, out=2, hidden=128, n_layers=15):
        super().__init__()
        self.node_norm = Normaliser(node_in)
        self.edge_norm = Normaliser(edge_in)
        self.out_norm = Normaliser(out)
        self.node_encoder = mlp(node_in, hidden, hidden)
        self.edge_encoder = mlp(edge_in, hidden, hidden)
        self.processors = nn.ModuleList(ProcessorLayer(hidden) for _ in range(n_layers))
        self.decoder = mlp(hidden, out, hidden, layer_norm=False)

    def forward(self, data):
        x = self.node_encoder(self.node_norm(data.x))
        e = self.edge_encoder(self.edge_norm(data.edge_attr))
        for layer in self.processors:
            x, e = layer(x, data.edge_index, e)
        return self.decoder(x)

    def free_mask(self, data):
        types = data.x[:, 2 : 2 + NODE_TYPES].argmax(1)
        return (types == NODE_NORMAL) | (types == NODE_OUTFLOW)

    def loss(self, data):
        prediction = self(data)
        target = self.out_norm(data.y)
        mask = self.free_mask(data)
        return ((prediction[mask] - target[mask]) ** 2).mean()
