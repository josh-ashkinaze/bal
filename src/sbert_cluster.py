from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

class SBERTClusterer():
    """
    A class for clustering texts using SBERT embeddings and KMeans.
    Args:
        model: SBERT model for text embeddings
        embeddings: Generated embeddings for input texts
        best_k: Best number of clusters found
        labels: Cluster assignments for texts
        scores: Silhouette scores for different k values
        centroids: Cluster centroids in embedding space
        centroid_texts: Text closest to each centroid
    Example:
        clusterer = SBERTClusterer()
        texts = ["text1", "text2", "text3"]
        clusterer.find_k(texts, k_min=2, k_max=3)
        df = clusterer.apply_clustering(texts, k=2)
        print(clusterer.centroids)  # Access centroids
        print(clusterer.centroid_texts)  # Access centroid texts
    """

    def __init__(self, sbert_model="all-MiniLM-L6-v2"):
        """Initialize with specified SBERT model."""
        self.model = SentenceTransformer(sbert_model)
        self.embeddings = None
        self.best_k = None
        self.labels = None
        self.scores = None
        self.centroids = None
        self.centroid_texts = None

    def _embed_texts(self, texts):
        """Generate embeddings for input texts."""
        self.embeddings = self.model.encode(texts)
        return self.embeddings

    def _find_centroid_texts(self, texts):
        """
        Find the text closest to each cluster centroid.
        Args:
            texts: List of original texts
        Returns:
            List of texts closest to each centroid
        """
        from scipy.spatial.distance import cdist

        centroid_texts = []
        for i, centroid in enumerate(self.centroids):
            # Find indices of texts in this cluster
            cluster_indices = np.where(self.labels == i)[0]
            cluster_embeddings = self.embeddings[cluster_indices]

            # Calculate distances from centroid to all texts in cluster
            distances = cdist([centroid], cluster_embeddings, metric='cosine')[0]

            # Find the closest text
            closest_idx = cluster_indices[np.argmin(distances)]
            centroid_texts.append(texts[closest_idx])

        return centroid_texts

    def find_k(self, texts, k_min=2, k_max=10, plot=True):
        """
        Find optimal k using silhouette scores.
        Args:
            texts: List of strings to cluster
            k_min: Minimum number of clusters to try
            k_max: Maximum number of clusters to try
            plot: Whether to plot silhouette scores
        Returns:
            Best k value found
        """
        self._embed_texts(texts)
        scores = []
        k_values = range(k_min, min(len(texts), k_max + 1))

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.embeddings)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(self.embeddings, labels)
                scores.append(score)
            else:
                scores.append(-1)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, scores, "bo-")
            plt.xlabel("K (Number of Clusters)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Scores by Number of Clusters")
            plt.grid(True)
            plt.show()

        self.best_k = k_values[np.argmax(scores)]
        self.scores = scores
        print(f"Best k: {self.best_k} with silhouette score: {max(scores):.3f}")
        logging.info(f"Best k: {self.best_k} with silhouette score: {max(scores):.3f}")
        return self.best_k

    def apply_clustering(self, texts, k=None):
        """
        Apply KMeans clustering with specified k.
        Args:
            texts: List of strings to cluster
            k: Number of clusters (uses best_k if None)
        Returns:
            DataFrame with columns [text, cluster, centroid_text]
        """
        if self.embeddings is None:
            self._embed_texts(texts)

        if k is None:
            if self.best_k is None:
                raise ValueError("Must either specify k or run find_k first")
            k = self.best_k

        kmeans = KMeans(n_clusters=k, random_state=42)
        self.labels = kmeans.fit_predict(self.embeddings)

        # Store centroids
        self.centroids = kmeans.cluster_centers_

        # Find centroid texts
        self.centroid_texts = self._find_centroid_texts(texts)

        # Create DataFrame with centroid text for each row
        centroid_text_map = {i: self.centroid_texts[i] for i in range(k)}
        df = pd.DataFrame({
            "text": texts,
            "cluster": self.labels,
            "centroid_text": [centroid_text_map[label] for label in self.labels]
        })

        for cluster in sorted(df["cluster"].unique()):
            cluster_texts = df[df["cluster"] == cluster]["text"].tolist()
            centroid_text = self.centroid_texts[cluster]
            print(f"\nCluster {cluster} (Centroid: '{centroid_text}'):")
            for text in cluster_texts:
                print(f" - {text}\n")
            logging.info(f"\nCluster {cluster} (Centroid: '{centroid_text}'):")
            logging.info(str(cluster_texts))

        return df

    def get_cluster_summary(self):
        """
        Get a summary of clusters with their centroids.
        Returns:
            DataFrame with cluster info and centroid texts
        """
        if self.centroids is None or self.centroid_texts is None:
            raise ValueError("Must run apply_clustering first")

        cluster_sizes = np.bincount(self.labels)
        summary_df = pd.DataFrame({
            "cluster": range(len(self.centroids)),
            "centroid_text": self.centroid_texts,
            "size": cluster_sizes
        })

        return summary_df