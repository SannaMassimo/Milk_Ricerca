import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

""" Fit and Predict Clusters with Time Series k Means"""

class ClusterGeneration():
    def __init__(self, default=True, cluster_path=None): # default define if cluster must be loaded or generated
        self.num_clusters = None
        if default:
            if cluster_path is None:
                raise Exception("cluster_path is required")
            else:
                saved_data = np.load(cluster_path)
                self.cluster_mapping = dict(zip(saved_data['cow_ids'], saved_data['clusters']))
                self.num_clusters = len(np.unique(saved_data['clusters']))
        else:
            self.cluster_mapping = None

    def fit_predict(self, n_clusters: int, data: pd.DataFrame, random_state=None, verbose = 0, n_jobs = -1):
        if random_state is None:
            model = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw')
        else:
            model = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', random_state=random_state, verbose=verbose, n_jobs=n_jobs)
        self.num_clusters = n_clusters

        cow_ids = data['id_cow'].unique()
        prod_series_data = [data[data['id_cow'] == cow_id]['tot_prod'].values for cow_id in cow_ids]

        prod_series_dataset = to_time_series_dataset(prod_series_data)
        
        scaler = TimeSeriesScalerMeanVariance()
        prod_series_dataset_scaled = scaler.fit_transform(prod_series_dataset)

        predicted_clusters = model.fit_predict(prod_series_dataset_scaled)

        self.cluster_mapping = dict(zip(cow_ids, predicted_clusters))

        return self.cluster_mapping # numpy-format output cluster

    def save_clusters(self, ouput_path: str = "clusters.npy"): # this save your clusters in numpy format 
        if self.cluster is None:
            raise Exception("Are you sure you compute clusters?")
        else:
            cow_ids = list(self.cluster_mapping.keys())
            clusters = list(self.cluster_mapping.values())
            np.savez(ouput_path, cow_ids=cow_ids, clusters=clusters)
            return print("File Saved!")

    def print_clusters(self, data): # this plot all the clusters showing the average production of each 
        if 'cluster' not in data.columns:
            raise ValueError("La colonna 'cluster' non è presente nel DataFrame 'data'.")
        if data['cluster'].isnull().all():
            raise ValueError("Attenzione: La colonna 'cluster' contiene solo valori NaN.")

        avg_prod_per_cluster_date = data.dropna(subset=['cluster']).groupby(['date', 'cluster'])['tot_prod'].mean()
        avg_prod_unstacked = avg_prod_per_cluster_date.unstack(level='cluster')

        fig, ax = plt.subplots(figsize=(18, 8))
        
        for i, cluster_id in enumerate(avg_prod_unstacked.columns):
            ax.plot(avg_prod_unstacked.index.to_numpy(), avg_prod_unstacked[cluster_id].to_numpy(), label=f'Cluster {int(cluster_id)}')

        ax.set_xlabel("Data")
        ax.set_ylabel("Produzione Media Giornaliera (kg)")
        ax.set_title("Andamento Produzione Media di Latte per Cluster nel Tempo")
        ax.legend(title="Cluster")
        ax.grid(True)
        plt.tight_layout()

        plt.show()
        
    # TODO: Define metrics function







