import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import os

df = pd.read_csv("heart_disease_data.csv")
print(f"✅ Dataset loaded with shape: {df.shape}")

df = df.drop_duplicates()
df = df.dropna()
print(f"✅ Cleaned dataset shape: {df.shape}")

numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
for col in numeric_cols:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
print("✅ Applied min-max normalization.")

for col in numeric_cols:
    df[col] = pd.qcut(df[col], 3, labels=False)
print("✅ Discretized numeric columns.")

model = BayesianNetwork([
    ("age", "fbs"),
    ("fbs", "target"),
    ("target", "chol"),
    ("target", "thalach")
])
print("✅ Bayesian Network structure defined (assignment version).")

model.fit(df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
print("✅ Model fitted with Bayesian Estimator.")

model.check_model()
print("✅ Model check passed.")

inference = VariableElimination(model)

result1 = inference.query(variables=["target"])
print("\nPosterior of 'target':")
print(result1)

result2 = inference.query(variables=["chol"], evidence={"target": 1})
print("\nP(chol | target=1):")
print(result2)

with open("inference_result.stxt", "w") as f:
    f.write("Posterior of 'target':\n")
    f.write(str(result1) + "\n\n")
    f.write("P(chol | target=1):\n")
    f.write(str(result2) + "\n")
print("✅ Inference results written to 'inference_result.stxt'")

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

os.makedirs("visualizations", exist_ok=True)

# save the PNG
plt.savefig("visualizations/bayesian_network.png", dpi=300)
plt.show()

print("✅ Bayesian Network diagram saved to visualizations/bayesian_network.png")
