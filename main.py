from data_loading import load_data, merge_with_cluster
from clustering import ClusterGeneration
from training import TrainingModel

path = "./data/AMerged.csv"
model_path = "./data/strangeModel"
cluster_path = "./data/clusters4.npy"

def instantiate_data(generateClusters: bool = False):
    if generateClusters:
        data = load_data(path, cluster=False)

        generator = ClusterGeneration(default=False)
        generator.fit_predict(n_clusters=6, data=data)
        generator.save_clusters(ouput_path=cluster_path)

        data = merge_with_cluster(data, cluster_path=cluster_path)
    else:
        data = load_data(path, cluster=True, cluster_path=cluster_path)

        generator = ClusterGeneration(default=True, cluster_path=cluster_path) # questo lo facciamo se magari ci interessa plottare cose etc
    return data, generator

def load_model(train: bool = False):
    if train:
        model = TrainingModel()
        model.train(data, verbose=True, random_state=42)
    else:
        model = TrainingModel(load=True, model_path=model_path, num_clusters=generator.num_clusters)    
    return model

if __name__ == "__main__":
    data, generator = instantiate_data(generateClusters=False)

    model = load_model(train=False)

    cow_id = "AZ10_481"
    model.plot_cow(data, cow_id)
