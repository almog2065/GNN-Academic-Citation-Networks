import csv

from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pickle
import networkx as nx
import numpy as np

from dataset import HW3Dataset
from torch_geometric.nn import GATConv, GCNConv, GINConv, ChebConv, SAGEConv, GraphConv, DNAConv, ARMAConv, APPNP, \
    TransformerConv, TAGConv

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


class MixedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MixedModel, self).__init__()
        self.gat_conv = GATConv(input_dim, hidden_dim)
        self.gin_conv = GINConv(
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.gcn_conv = GCNConv(input_dim, hidden_dim)
        self.cheb_conv = ChebConv(input_dim, hidden_dim, 2)
        self.sage_conv = SAGEConv(input_dim, hidden_dim)
        self.g_conv = GraphConv(input_dim, hidden_dim)
        self.dna_conv = TransformerConv(input_dim, hidden_dim)
        self.arma_conv = ARMAConv(input_dim, hidden_dim)
        self.rgcn_conv = APPNP(input_dim, hidden_dim)
        self.tag_conv = TAGConv(input_dim, hidden_dim)

        self.lin = nn.Linear(3 * hidden_dim, 2 * hidden_dim)
        self.lin2 = nn.Linear(2 * hidden_dim, 1 * hidden_dim)
        self.lin3 = nn.Linear(1 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.4)
        self.batch_norm = nn.BatchNorm1d(3 * hidden_dim)

        # Initialize linear layer parameters
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, edge_index):
        x_gat = self.gat_conv(x, edge_index)
        x_gin = self.gin_conv(x, edge_index)
        x_gcn = self.gcn_conv(x, edge_index)
        x_cheb = self.cheb_conv(x, edge_index)
        x_sage = self.sage_conv(x, edge_index)
        x_g = self.g_conv(x, edge_index)
        # x_dna = self.dna_conv(x, edge_index)
        # x_arma = self.arma_conv(x, edge_index)
        # x_rgcn = self.rgcn_conv(x, edge_index)
        # x_tag = self.tag_conv(x, edge_index)

        x = torch.cat([x_gat, x_gcn, x_sage], dim=1)
        x = self.batch_norm(x)
        x = self.dropout(F.relu(x))
        x = self.lin(x)
        x = self.dropout(F.relu(x))
        x = self.lin2(x)
        x = self.dropout(F.relu(x))
        x = self.lin3(x)

        return x


def train_and_eval(model, optimizer, criterion, x, edge_index, y):
    acc_list, loss_list = [], []
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        output = model(x, edge_index)
        loss = criterion(output[data.train_mask], y[data.train_mask].squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        model.eval()
        with torch.no_grad():
            predicted = model(x, edge_index).max(dim=1)[1]
        correct = predicted[data.val_mask] == y[data.val_mask]
        acc = int(correct.sum()) / len(correct)
        # acc_list.append(acc)
        loss_list.append(loss.item())

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Test Accuracy: {acc * 100:.4f}%")
        if epoch+1 % 100 == 0:
            acc_list.append(acc)
            torch.save(model.state_dict(), f'model_{epoch}.pkl')
            with open(f'model_{epoch}.pkl', 'wb') as f:
                pickle.dump(model, f)

    return acc_list, loss_list, model


def train(data):
    x, edge_index, y = data.x, data.edge_index, data.y.squeeze(1)
    # node_year = data.node_year.to(x.dtype)
    # x = torch.cat((x, node_year), dim=1)
    N = data.num_features

    # model = GAT(N, hidden_dim=100, output_dim=dataset.num_classes, num_heads=3)
    # model = GCN(N, hidden_dim=300, output_dim=dataset.num_classes)
    # model = GIN(N, hidden_dim=300, output_dim=dataset.num_classes, num_layers=3)
    # model = FusedGNN(N, hidden_dim=300, num_classes=dataset.num_classes)
    model = MixedModel(N, hidden_dim=300, output_dim=dataset.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    acc_list, loss_list, model = train_and_eval(model, optimizer, criterion, x, edge_index, y)
    # plt.switch_backend('agg')
    # plt.plot([i + 1 for i in range(len(acc_list))], acc_list)
    # plt.title("MixedModel with year Validation Accuracy each epoch")
    # plt.savefig('MixedModel_year_acc.png')
    # plt.show()
    # plt.plot([i + 1 for i in range(len(loss_list))], loss_list)
    # plt.title("GIN Train Loss each epoch")
    # plt.savefig('GIN_loss.png')
    # plt.show()
    print(acc_list)

    torch.save(model.state_dict(), 'model.pkl')
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)


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


def predict(data):
    x, edge_index = data.x, data.edge_index
    model = MixedModel(data.num_features, hidden_dim=300, output_dim=dataset.num_classes)
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    model.load_state_dict(loaded_model.state_dict())

    model.eval()
    with torch.no_grad():
        predictions = model(x, edge_index).max(dim=1)[1]

    csv_file = 'prediction.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['Idx', 'Prediction'])

        for i, prediction in enumerate(predictions):
            writer.writerow([i, int(prediction)])


if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3')
    data = dataset[0]

    # analysis(data)

    # model = train(data)

    predict(data)
