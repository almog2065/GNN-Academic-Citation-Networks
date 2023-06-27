import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pickle
import networkx as nx
import numpy as np
import seaborn as sns
import scipy.sparse as sp

from dataset import HW3Dataset
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_add_pool, global_max_pool
from torch_geometric.utils import add_self_loops
import torch_geometric.nn as gnn
import torch_geometric.utils as gutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, output_dim, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GIN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.lin(x)
        return x


class MixedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MixedModel, self).__init__()
        self.gat_conv = GATConv(input_dim, hidden_dim)
        self.gin_conv = GINConv(
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.gcn_conv = GCNConv(input_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim * 3, output_dim)

    def forward(self, x, edge_index):
        x_gat = self.gat_conv(x, edge_index)
        x_gin = self.gin_conv(x, edge_index)
        x_gcn = self.gcn_conv(x, edge_index)

        x = torch.cat([x_gat, x_gin, x_gcn], dim=1)  # Concatenate the outputs from different convolutions
        x = F.relu(x)
        x = self.lin(x)

        return x


class MixedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MixedModel, self).__init__()
        self.gat_conv = GATConv(input_dim, hidden_dim)
        self.gin_conv = GINConv(
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.gcn_conv = GCNConv(input_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim * 3, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 3)

        # Initialize linear layer parameters
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, edge_index):
        x_gat = self.gat_conv(x, edge_index)
        x_gin = self.gin_conv(x, edge_index)
        x_gcn = self.gcn_conv(x, edge_index)

        x = torch.cat([x_gat, x_gin, x_gcn], dim=1)  # Concatenate the outputs from different convolutions
        x = self.batch_norm(x)  # Apply batch normalization
        x = self.dropout(F.relu(x))  # Apply dropout and activation function
        x = self.lin(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x


class FusedGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FusedGNN, self).__init__()
        self.gat_conv = GATConv(input_dim, hidden_dim)
        self.gat2_conv = GATConv(hidden_dim, hidden_dim)
        self.gin_conv = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x_gat = self.gat_conv(x, edge_index)
        # x_gin = self.gin_conv(x_gat, edge_index)
        x_gcn = self.gcn_conv(x_gat, edge_index)
        x_gat2 = self.gat2_conv(x_gcn, edge_index)

        x = F.relu(x_gat2)
        x = self.lin(x)

        return x


def train_and_eval(model, optimizer, criterion, x, edge_index):
    acc_list, loss_list = [], []
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        output = model(x, edge_index)
        loss = criterion(output[data.train_mask], y[data.train_mask].squeeze())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predicted = model(x, edge_index).max(dim=1)[1]
        correct = predicted[data.val_mask] == y[data.val_mask]
        acc = int(correct.sum()) / len(correct)
        acc_list.append(acc)
        loss_list.append(loss.item())

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Test Accuracy: {acc * 100:.4f}%")

    with open('MixedModel.pkl', 'wb') as f:
        pickle.dump(model, f)
    return acc_list, loss_list


def prediction(data):
    x, edge_index, y = data.x, data.edge_index, data.y.squeeze(1)
    node_year = data.node_year.to(x.dtype)
    x = torch.cat((x, node_year), dim=1)
    N = data.num_features + 1

    # model = GAT(N, hidden_dim=100, output_dim=dataset.num_classes, num_heads=3)
    # model = GCN(N, hidden_dim=300, output_dim=dataset.num_classes)
    # model = GIN(N, hidden_dim=300, output_dim=dataset.num_classes, num_layers=3)
    # model = FusedGNN(N, hidden_dim=300, num_classes=dataset.num_classes)
    model = MixedModel(N, hidden_dim=400, output_dim=dataset.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    acc_list, loss_list = train_and_eval(model, optimizer, criterion, x, edge_index)
    plt.switch_backend('agg')
    plt.plot([i + 1 for i in range(len(acc_list))], acc_list)
    plt.title("MixedModel with year Validation Accuracy each epoch")
    plt.savefig('MixedModel_year_acc.png')
    plt.show()
    # plt.plot([i + 1 for i in range(len(loss_list))], loss_list)
    # plt.title("GIN Train Loss each epoch")
    # plt.savefig('GIN_loss.png')
    # plt.show()


def analysis(data):
    x, edge_index, y = data.x, data.edge_index, data.y.squeeze(1)
    node_year = data.node_year.to(x.dtype)

    G = nx.Graph()

    for node_id, features in enumerate(x):
        node_attrs = {'year': node_year[node_id].item()}
        G.add_node(node_id, **node_attrs)

    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    average_degree = sum(dict(G.degree()).values()) / num_nodes
    max_degree = max(dict(G.degree()).values())
    min_degree = min(dict(G.degree()).values())
    min_year = min(node_year)
    max_year = max(node_year)

    print("Graph Analysis Results:")
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)
    print("Average degree:", average_degree)
    print("Max degree:", max_degree)
    print("Min degree:", min_degree)
    print("Minimum year:", int(min_year[0]))
    print("Maximum year:", int(max_year[0]))

    years = data.node_year.numpy().T
    labels = data.y.squeeze(1).numpy()
    correlation = np.corrcoef(years, labels)[0, 1]
    print("Correlation coefficient of labels and year:", correlation)

    counts = {label.item(): 0 for label in set(y)}
    for label in y:
        counts[label.item()] += 1

    plt.bar(list(counts.keys()), list(counts.values()))
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Labels Histogram')
    plt.show()

    degrees = dict(G.degree()).values()
    plt.scatter(range(len(degrees)), degrees)
    plt.title("Degree Scatter Plot")
    plt.xlabel("Node index")
    plt.ylabel("Degree")
    plt.show()


if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3')
    data = dataset[0]

    # analysis(data)

    prediction(data)
