import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

df = pd.read_csv("Mall_Customers.csv")

df["Gender"]= df["Gender"].map({"Female":0, "Male":1})  
features = df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]]
x = StandardScaler().fit_transform(features)
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(x)
tsne = TSNE(n_components=2, random_state=42)
x_embedded = tsne.fit_transform(x)
plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=df["Cluster"], cmap='viridis')
plt.title("Customer Segments Visualization using t-SNE")
plt.show()