from data_loading import load_data, merge_with_cluster
from clustering import ClusterGeneration
from training import TrainingModel
import shutil
import yaml
import os

config_path = 'config.yaml'

data_path = "./data/AMerged.csv"
model_path = "./data/test1"
cluster_path = "./data/clusters4.npy"

def load_conf():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_path = config['paths']['input_data']
    cluster_path = config['paths']['cluster_file']
    model_path = config['paths']['model_output_dir']
    os.makedirs(model_path, exist_ok=True)

    print("Caricamento dati...\n")
    
    data = load_data(config['paths']['input_data'], config['clustering']['perform_clustering'], cluster_path)

    if config['clustering']['perform_clustering']:
        print(f"Generazione di {config['clustering']['n_clusters']} cluster...")
        
        generator = ClusterGeneration(default=False)
        generator.fit_predict(n_clusters=config['clustering']['n_clusters'], data=data, random_state = config['clustering']['random_state'], verbose = config['verbose'])
        generator.save_clusters(ouput_path=cluster_path)
        
        print(f"Cluster generati e salvati in {cluster_path}\n")

        data = merge_with_cluster(data, cluster_path)

    else: 
        generator = ClusterGeneration(default=True, cluster_path=cluster_path)


    model = TrainingModel(config)
    
    if config['training']['perform_training']:
        print("Inizio training")

        model.train(config, data)
        
        print("Training Completato")

        output_config_path = os.path.join(model_path, 'config.yaml')
        shutil.copy(config_path, output_config_path)

        with open(output_config_path, 'r') as f:
            run_config = yaml.safe_load(f)

        run_config['training']['perform_training'] = False

        with open(output_config_path, 'w') as f:
            yaml.dump(run_config, f, default_flow_style=False)


    return data, model, generator
        

if __name__ == "__main__":
    data, model, generator = load_conf()

    cow_id = "AZ10_481"
    model.plot_cow(data, cow_id)
