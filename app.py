import streamlit as st
import pandas as pd
import torch
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Define GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Streamlit app
st.set_page_config(layout="wide")
st.title("Housing Price Forecast with GNN and Machine Learning")

# Sidebar for parameter adjustments
st.sidebar.write("## Parameters")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.sidebar.write("### Dataset Columns:")
    st.sidebar.write(df.columns.tolist())

    # Normalize numerical attributes
    scaler = MinMaxScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    normalized_df = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns)

    # Attribute ranking by user
    st.sidebar.write("### Rank Attributes")
    rankings = {}
    for col in numeric_columns:
        rankings[col] = st.sidebar.slider(f"Rank for {col}", min_value=1, max_value=10, value=5)

    # Ensure the length of y_numeric matches the number of rows in X_numeric
    y_numeric = np.array([rankings[col] for col in numeric_columns])
    y_numeric = np.tile(y_numeric, (len(normalized_df), 1)).mean(axis=1)

    # Calculate numerical coefficients using machine learning
    X_numeric = normalized_df
    rf_model = RandomForestRegressor()
    rf_model.fit(X_numeric, y_numeric)
    numeric_coefficients = rf_model.feature_importances_

    # Prepare data for GNN (if textual columns exist)
    textual_columns = [col for col in df.columns if col not in numeric_columns]
    if textual_columns:
        st.sidebar.write("### Textual Columns:")
        st.sidebar.write(textual_columns)

        # Example: Convert textual columns into node features using embeddings (not implemented in full detail here)
        node_features = torch.randn(len(df), 16)  # Example random node features
        edge_index = torch.tensor([[i, j] for i in range(len(df)) for j in range(len(df))], dtype=torch.long).t()

        gnn_model = GNNModel(num_features=node_features.shape[1], num_classes=1)
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(100):
            gnn_model.train()
            optimizer.zero_grad()
            predictions = gnn_model(node_features, edge_index)
            textual_contributions = predictions.flatten().detach().numpy()
    else:
        textual_contributions = np.zeros(len(df))

    # Compute utility values
    utility_values = np.zeros(len(df))
    for i, col in enumerate(numeric_columns):
        utility_values += numeric_coefficients[i] * normalized_df[col]
    utility_values += textual_contributions

    # Select top 10 utility values as optimal subsets
    df['Utility'] = utility_values
    optimal_subsets = df.nlargest(10, 'Utility')
    optimal_indices = optimal_subsets.index.tolist()

    # Display results
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("### Optimal Subset Indices")
        st.write(optimal_indices)

    with col2:
        st.write("### Utility Function")
        utility_function = " + ".join([f"{coeff:.4f} * {col}" for coeff, col in zip(numeric_coefficients, numeric_columns)])
        if textual_columns:
            utility_function += " + Textual Contributions"
        st.write(f"y = {utility_function}")

        st.write("### Optimal Subsets")
        st.write(optimal_subsets)

else:
    st.write("Please upload a dataset to proceed.")
