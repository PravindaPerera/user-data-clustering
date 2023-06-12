from sklearn.cluster import KMeans
import numpy as np

# Assume you have already performed K-Means clustering and obtained the cluster centers

# Sample data points and cluster centers
data_points = np.array([[2, 3], [5, 6], [1, 4], [6, 2], [3, 5]])
cluster_centers = np.array([[2, 2], [5, 5]])

# Initialize K-Means model
kmeans = KMeans(n_clusters=2)
kmeans.cluster_centers_ = cluster_centers

# Fit K-Means model to data points
kmeans.fit(data_points)

# New record to be classified
new_record = np.array([[6, 5]])

# Predict the cluster for the new record
predicted_cluster = kmeans.predict(new_record)

print("Predicted Cluster:", predicted_cluster)
