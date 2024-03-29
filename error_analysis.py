import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")

# load the data
data = pd.read_csv("error_analysis.csv")

"""
data columns:
"pred", "label", "knn1", "knn2", "knn3", "knn4", "knn5",
"dist1", "dist2", "dist3", "dist4", "dist5"
"""

# confusion matrix
cm = confusion_matrix(data["label"], data["pred"], labels=list(range(103)), normalize="true")
plt.figure(figsize=(20, 20))
plt.imshow(cm, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(103), rotation=90)
plt.yticks(range(103))
plt.tight_layout()
plt.savefig("confusion_matrix.png")
