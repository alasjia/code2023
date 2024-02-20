import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, adj_matrix, node_features):
        # adj_matrix: Adjacency matrix of the graph (in sparse format or as a tensor)
        # node_features: Input features of the nodes
        
        # Perform graph convolution (similar to GCN)
        support = self.linear(node_features)
        output = torch.spmm(adj_matrix, support)  # Sparse matrix multiplication
        
        return output

class GraphNeuralNetwork(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.gcn1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gcn2 = GraphConvolutionLayer(hidden_dim, output_dim)
    
    def forward(self, adj_matrix, node_features):
        # adj_matrix: Adjacency matrix of the graph (in sparse format or as a tensor)
        # node_features: Input features of the nodes
        
        # Apply the first GCN layer
        hidden = F.relu(self.gcn1(adj_matrix, node_features))
        
        # Apply the second GCN layer
        output = self.gcn2(adj_matrix, hidden)
        
        return output

# Set a random seed for reproducibility
random.seed(42)

# Generate the adjacency matrix (undirected graph with 4 nodes)
# Let's define the graph as follows:
# Node 0 is connected to nodes 1 and 2.
# Node 1 is connected to nodes 0 and 3.
# Node 2 is connected to nodes 0 and 3.
# Node 3 is connected to nodes 1 and 2.
adj_matrix = torch.tensor([[0, 1, 1, 0],
                           [1, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0]], dtype=torch.float)

# Generate node features (random features for simplicity)
num_nodes = adj_matrix.size(0)
num_features = 2
node_features = torch.rand(num_nodes, num_features)

# Generate node labels (binary classification, two classes)
node_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

# print("Adjacency Matrix:")
# print(adj_matrix)
# print("\nNode Features:")
# print(node_features)
# print("\nNode Labels:")
# print(node_labels)

# Example usage:
# Assuming you have the adjacency matrix (adj_matrix) and node features (node_features) of the graph
# Create the GNN model
num_nodes = len(node_features)
input_dim = node_features.shape[1]
hidden_dim = 64
output_dim = 10

model = GraphNeuralNetwork(num_nodes, input_dim, hidden_dim, output_dim)

# Forward pass
output = model(adj_matrix, node_features)
print("\noutput:")
print(output)