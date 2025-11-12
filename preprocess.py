import os
import pandas as pd
from sklearn.datasets import load_iris

# ✅ Ensure output directory exists
os.makedirs("data", exist_ok=True)

# ✅ Load dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["target"] = iris.target

# ✅ Save preprocessed data
output_path = "data/preprocessed.csv"
data.to_csv(output_path, index=False)

print(f"✅ Data preprocessed and saved to {output_path}")
