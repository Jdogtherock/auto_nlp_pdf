# kmeans clustering methods
# will do kmeans from 2 to 12 clusters, and generate the silhoulette score, WCSS, davies-bouldin, elbow plot, and cluster scatter plot
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io

def kmeans_clustering(data, start=2, end=12):
    wcss_values = []
    silhouette_scores = []

    num_clusters = range(start, end+1)

    for k in num_clusters:
        kmeans = KMeans(n_clusters=k)
        predicted_labels = kmeans.fit_predict(data)

        # Compute metrics for current number of clusters
        wcss_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, predicted_labels))

    # Elbow plot
    plt.figure()
    plt.plot(num_clusters, wcss_values, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    elbow_plot = buf

    # PCA scatter plot for optimal cluster (based on silhouette score)
    optimal_k = num_clusters[np.argmax(silhouette_scores)]
    kmeans = KMeans(n_clusters=optimal_k)
    predicted_labels = kmeans.fit_predict(data)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    plt.figure()
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=predicted_labels, cmap='rainbow', edgecolors='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='X')
    plt.title(f'PCA Cluster Scatter Plot (Optimal {optimal_k} clusters)')
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png")
    buf2.seek(0)
    pca_plot = buf2

    # Metrics summary for optimal cluster
    optimal_metrics_summary = {
        "Clusters": optimal_k,
        "WCSS": wcss_values[np.argmax(silhouette_scores)],
        "Silhouette Score": np.max(silhouette_scores),
        "Davies Bouldin Index": davies_bouldin_score(data, predicted_labels)
    }

    return elbow_plot, pca_plot, optimal_metrics_summary

# Sample data
from sklearn.datasets import make_blobs
data, _ = make_blobs(n_samples=300, centers=6, cluster_std=0.60, random_state=0)

elbow_plot, pca_plot, optimal_metrics_summary = kmeans_clustering(data)

# In your Streamlit application, you can then use:
# st.image(elbow_plot)
# st.image(pca_plot)
# st.write(optimal_metrics_summary)
