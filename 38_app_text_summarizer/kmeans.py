import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 🎯 Step 1: Generate some sample data
# Think of these as customers with 2 features: spending vs visits
data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# 📊 Step 2: Visualize the ungrouped data
plt.scatter(data[:, 0], data[:, 1], s=30)
plt.title("Unclustered Data")
plt.show()

# 🤖 Step 3: Apply KMeans to group the data into 4 clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
labels = kmeans.labels_  # Group assignments for each point
centers = kmeans.cluster_centers_  # The cluster centers

# 🎨 Step 4: Visualize the grouped (clustered) data
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='X')  # Centroids
plt.title("Clustered Data with K-Means")
plt.show()
