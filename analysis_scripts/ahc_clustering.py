import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import networkx as nx
import numpy as np

# Load your graph
G = nx.read_graphml("path_to_your_graphml_file.graphml")

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)  # Assuming we want a single feature for clustering

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)  # Same size as hidden for simplicity

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))

# Define Self-Attention model
class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=1, batch_first=True)
        self.fc = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        # Linear projection for attention
        x_proj = self.fc(x)
        # Apply self-attention
        attn_output, _ = self.attention(x_proj, x_proj, x_proj)
        return attn_output

# Convert to PyTorch Geometric Data
data = from_networkx(G)
data.x = torch.tensor([G.nodes[node]['feature'] for node in G.nodes()], dtype=torch.float)  # Replace 'feature' with actual feature name

# Initialize GCN
model = GCN(data.num_node_features, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Assume we have node features (x) and labels (y)
# Train the GCN model
# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     loss = some_loss_function(out, data.y)  # Define the appropriate loss function
#     loss.backward()
#     optimizer.step()

# Generate embeddings
model.eval()
with torch.no_grad():
    gcn_embeddings = model(data.x, data.edge_index)

# LSTM Training (Placeholder)
# Define your time sequences for LSTM
time_sequences = torch.randn((data.num_nodes, 10, 1))  # Randomly generated; replace with actual data
lstm_model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
# Train your LSTM model
# LSTM parameters
# input_size = 1  # Input size (the number of features per time step)
# hidden_size = 64  # The size of LSTM hidden states
# num_layers = 2  # The number of stacked LSTM layers
# output_size = 64  # The size of the output vector

# # Create TensorDataset and DataLoader
# dataset = TensorDataset(time_sequences)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Initialize LSTM model
# lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

# # Train the model
# epochs = 10
# for epoch in range(epochs):
#     for sequences_batch, in dataloader:
#         optimizer.zero_grad()
#         outputs = lstm_model(sequences_batch)
#         # Here we use outputs to predict the sequences themselves (like autoencoder)
#         loss = criterion(outputs, sequences_batch[:, -1, :])
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# # Generate LSTM vectors
# lstm_vectors = lstm_model(time_sequences)

# Generate LSTM vectors
with torch.no_grad():
    lstm_vectors = lstm_model(time_sequences)

# Combine node attributes with embeddings
node_attributes = torch.tensor([G.nodes[node]['attribute'] for node in G.nodes()], dtype=torch.float)  # Replace 'attribute' with actual attributes
combined_features = torch.cat((gcn_embeddings, lstm_vectors, node_attributes), dim=1)

# Apply self-attention
self_attention = SelfAttention(input_dim=combined_features.size(1), attention_dim=64)
attended_features = self_attention(combined_features)

# Find the best number of clusters based on Silhouette Coefficient
silhouette_scores = []
for k in range(2, 10):  # Assuming the range of k values to be between 2 and 10
    clustering_model = AgglomerativeClustering(n_clusters=k)
    clustering_model.fit(attended_features)
    silhouette_avg = silhouette_score(attended_features, clustering_model.labels_)
    silhouette_scores.append(silhouette_avg)

best_k = np.argmax(silhouette_scores) + 2  # Plus 2 because range starts at 2

# Perform clustering
clustering_model = AgglomerativeClustering(n_clusters=best_k)
cluster_labels = clustering_model.fit_predict(attended_features)

# Add cluster labels to nodes
for i, node in enumerate(G.nodes()):
    G.nodes[node]['cluster_label'] = cluster_labels[i]

# Export graph with cluster labels
nx.write_graphml(G, "/home/ta/gambling/0xc2a_Complete.graphml")
