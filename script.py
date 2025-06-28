import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import os

# Step 1: Load Dataset
df = pd.read_csv("heart_disease_data.csv")
print(f"✅ Dataset loaded with shape: {df.shape}")

# Step 2: Clean data
df = df.drop_duplicates()
df = df.dropna()
print(f"✅ Cleaned dataset shape: {df.shape}")

# Step 3: Min-max normalization on numeric columns
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
for col in numeric_cols:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
print("✅ Applied min-max normalization.")

# Step 4: Discretize normalized columns (3 bins)
for col in numeric_cols:
    df[col] = pd.qcut(df[col], 3, labels=False)
print("✅ Discretized numeric columns.")

# Step 5: Define Bayesian Network structure as per assignment
model = BayesianNetwork([
    ("age", "fbs"),
    ("fbs", "target"),
    ("target", "chol"),
    ("target", "thalach")
])
print("✅ Bayesian Network structure defined (assignment version).")

# Step 6: Fit the model
model.fit(df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
print("✅ Model fitted with Bayesian Estimator.")

# Step 7: Check the model
model.check_model()
print("✅ Model check passed.")

# Step 8: Inference
inference = VariableElimination(model)

# Posterior of 'target'
result1 = inference.query(variables=["target"])
print("\nPosterior of 'target':")
print(result1)

# Posterior of 'chol' given target = 1
result2 = inference.query(variables=["chol"], evidence={"target": 1})
print("\nP(chol | target=1):")
print(result2)

# Save inference results
with open("inference_result.stxt", "w") as f:
    f.write("Posterior of 'target':\n")
    f.write(str(result1) + "\n\n")
    f.write("P(chol | target=1):\n")
    f.write(str(result2) + "\n")
print("✅ Inference results written to 'inference_result.stxt'")

# Step 9: Visualize and save Bayesian Network graph
plt.figure(figsize=(10, 7))
G = nx.DiGraph(model.edges())

nx.draw(
    G,
    pos=nx.spring_layout(G, seed=42),
    with_labels=True,
    node_size=3000,
    node_color="skyblue",
    font_size=10,
    font_weight="bold",
    edge_color="gray",
    arrows=True
)
plt.title("Bayesian Network: Heart Disease Assignment Structure", fontsize=14)

# ensure the visualizations folder exists
os.makedirs("visualizations", exist_ok=True)

# save the PNG
plt.savefig("visualizations/bayesian_network.png", dpi=300)
plt.show()

print("✅ Bayesian Network diagram saved to visualizations/bayesian_network.png")
