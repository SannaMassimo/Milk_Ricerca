from data_loading import load_data, merge_with_cluster
from utilsTraining import split_cows_by_id
from clustering import ClusterGeneration
from training import TrainingModel
from analysis import ModelAnalyzer
import shutil
import torch
import yaml
import os

config_path = 'experiments/run_01/config.yaml'

data_path = "./data/AMerged.csv"
model_path = "./data/test1"
cluster_path = "./data/clusters4.npy"

def load_conf():
    global config_path, model_path, cluster_path, config, data

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_path = config['paths']['input_data']
    cluster_path = config['paths']['cluster_file']
    model_path = config['paths']['model_output_dir']
    os.makedirs(model_path, exist_ok=True)

    print("Caricamento dati...\n")
    
    data = load_data(config['paths']['input_data'], not config['clustering']['perform_clustering'], cluster_path)

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
    config_path = input("Wich config file do you want to start? (example: \"experiments/run_01/config.yaml\")\n")
    
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    data, model, generator = load_conf()

    loop1 = True
    while(loop1):
        scelta = input("\nInsert: \n1: To Analyze the model\n2: To chosee a cow to analyze\n3: to exit\n")

        if scelta == "1":
            _ , global_test_data, _ = split_cows_by_id(data, random_state=config['training']['random_state'])
            
            analyzer = ModelAnalyzer(model_path=model_path, config=config)
            analyzer.analyze_performance(global_test_data)
            analyzer.analyze_feature_importance(global_test_data)

        elif scelta == "2":
            loop2 = True
            while(loop2):
                #cow_id = "AZ10_481"
                cow_id = input("Choose cow id to analyze (for example AZ10_481)\n")

                model.predict_cow(data, cow_id)

                loop2 = False

        elif scelta == "3":
            loop1 = False
