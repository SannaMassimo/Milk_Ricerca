import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import torch
import os

from utilsTraining import prepareData, EarlyStopping, prepare_sequences, set_seeds
 
""" Training of the data """

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout_rate: float):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TrainingModel:
    def __init__(self, config):
        set_seeds(config['training']['random_state'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_name = config['training']['target_name']
        self.features = config['features']
        self.num_layers = config['training']['model_params']['num_layers']
        self.hidden_size = config['training']['model_params']['hidden_size']
        self.dropout_rate = config['training']['model_params']['dropout_rate']
        self.sequence_length = config['training']['sequence_length']
        self.models = {}
        self.feature_scalers = {}
        self.target_scalers = {}
        self.model_path = config['paths']['model_output_dir']
        if config['training']['perform_training'] == False: # load is True means that we want to load the model, so we don't need to train it
            self.loadModel(model_path=self.model_path, num_clusters=config['clustering']['n_clusters'])
                
    def loadModel(self, model_path=None, num_clusters=None):
        if model_path is None:
            raise ValueError("model_path must be provided when load is True")
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist")
        if num_clusters is None:
            raise ValueError("num_clusters must be provided when load is True") 
        self.models = {}
        self.feature_scalers = {}
        self.target_scalers = {}
        for i in range(num_clusters):
            model_path_i = os.path.join(model_path, f'lstm_model_cluster_{i}.pt')
            if not os.path.exists(model_path_i):
                raise ValueError(f"Model path {model_path_i} does not exist")
            model = LSTMModel(input_size=len(self.features), hidden_size=self.hidden_size, num_layers=self.num_layers, dropout_rate=self.dropout_rate)
            model.load_state_dict(torch.load(model_path_i, map_location=self.device))
            model.eval()
            self.models[i] = model 
            
            feature_scaler_path = os.path.join(model_path, f'feature_scaler_cluster_{i}.pkl')
            target_scaler_path = os.path.join(model_path, f'target_scaler_cluster_{i}.pkl')

            if os.path.exists(feature_scaler_path) == False or os.path.exists(target_scaler_path) == False:
                raise Exception(f"Feature scaler or target scaler not found for cluster {i}")

            self.feature_scalers[i] = joblib.load(feature_scaler_path)
            self.target_scalers[i] = joblib.load(target_scaler_path)

    def train(self, config, data: pd.DataFrame, random_state = None):
        random_state = config['training']['random_state']
        epochs = config['training']['hyperparameters']['epochs']
        batch_size = config['training']['hyperparameters']['batch_size']
        lr = config['training']['hyperparameters']['learning_rate']
        test_size = config['training']['test_size']
        weight_decay_rate = config['training']['hyperparameters']['weight_decay']
        patience = config['training']['early_stopping']['patience']
        delta = config['training']['early_stopping']['delta']

        set_seeds(random_state)

        if 'cluster' not in data.columns:
            raise ValueError("Errore: il dataframe non risulta clusterizzato.")
        clusters = data['cluster'].unique()
        cluster_losses = {}

        models_dir = self.model_path
        os.makedirs(models_dir, exist_ok=True)
        
        self.models = {} # Resetta/Inizializza
        self.feature_scalers = {}
        self.target_scalers = {}

        for cluster in clusters: # Loop through each cluster
            print(f"Cluster: {cluster}")
            cluster_data = data[data['cluster'] == cluster].copy()
            if cluster_data['id_cow'].nunique() < 2 and len(cluster_data) < (self.sequence_length + 1) * 5: # Check se ci sono abbastanza dati per fare training/test sensato
                print(f"Cluster {cluster} ha pochi dati ({len(cluster_data)} righe). Salto.")
                continue
            
            train_loader, test_loader, f_scaler, t_scaler = prepareData(cluster_data, self.device, random_state, self.features, self.target_name, self.sequence_length, batch_size, test_size)
            self.feature_scalers[cluster] = f_scaler
            self.target_scalers[cluster] = t_scaler
            joblib.dump(f_scaler, os.path.join(models_dir, f'feature_scaler_cluster_{cluster}.pkl')) # save the scalers
            joblib.dump(t_scaler, os.path.join(models_dir, f'target_scaler_cluster_{cluster}.pkl')) 
            
            model = LSTMModel(input_size=len(self.features), hidden_size=self.hidden_size, num_layers=self.num_layers, dropout_rate=self.dropout_rate).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr, weight_decay=weight_decay_rate)
            model_path = os.path.join(models_dir, f'lstm_model_cluster_{cluster}.pt')
            early_stopping = EarlyStopping(patience=patience, path=model_path, delta=delta)

            train_losses = []
            val_losses = []
            # Training loop
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device).unsqueeze(1)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch_X.size(0)

                epoch_loss /= len(train_loader.dataset)
                train_losses.append(epoch_loss)

                model.eval()
                val_mse = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device).unsqueeze(1)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_mse += loss.item() * batch_X.size(0)

                val_mse /= len(test_loader.dataset)
                val_losses.append(val_mse)

                if (epoch + 1) % 5 == 0:
                    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Test Loss: {val_mse:.4f}')

                early_stopping(val_mse, model)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            self.models[cluster] = early_stopping.best_model
            print(f"Best MSE per cluster {cluster}: {-early_stopping.best_score:.4f}")
            
        self.model_path = models_dir

        print("\nLoss finale per ogni cluster:")
        for cl, loss in cluster_losses.items():
            print(f"Cluster {cl}: train loss:  {loss['train_loss']:.4f}; test loss: {loss['test_loss']:.4f}")


    def predict_cow(self, data: pd.DataFrame, cow_id: str):
        start_date = pd.to_datetime('2021-07-01')
        end_date = pd.to_datetime('2021-09-30')
        data2 = data[(data["date"] >= start_date) & (data["date"] <= end_date)]
        if cow_id not in data2['id_cow'].values:
            if cow_id not in data['id_cow'].values:
                raise Exception("Cow ID non trovato nei dati.")
        else:
            data = data2   

        cow_data_orig = data[data['id_cow'] == cow_id].copy().sort_values('date')
        
        cluster_id = int(cow_data_orig['cluster'].iloc[0])
        print(f"Mucca {cow_id} appartiene al cluster {cluster_id}.")

        cow_features_scaled_array = self.feature_scalers[cluster_id].transform(cow_data_orig[self.features])
        cow_target_scaled_array = self.target_scalers[cluster_id].transform(cow_data_orig[[self.target_name]]).flatten()

        cow_scaled_df = pd.DataFrame(cow_features_scaled_array, columns=self.features, index=cow_data_orig.index)
        cow_scaled_df[self.target_name] = cow_target_scaled_array
        cow_scaled_df['id_cow'] = cow_data_orig['id_cow']
        cow_scaled_df['date'] = cow_data_orig['date']

        cow_scaled_df = cow_scaled_df.dropna(subset=self.features + [self.target_name] + ['id_cow', 'date'])

        X_plot, y_plot_scaled = prepare_sequences(cow_scaled_df, self.features, self.target_name, self.sequence_length)
        X_plot_tensor = torch.tensor(X_plot, dtype=torch.float32).to(self.device)

        model_plot = LSTMModel(input_size=len(self.features), hidden_size=self.hidden_size, num_layers=self.num_layers, dropout_rate=self.dropout_rate).to(self.device)
        plot_model_path = os.path.join(self.model_path, f'lstm_model_cluster_{cluster_id}.pt')
        model_plot.load_state_dict(torch.load(plot_model_path, map_location=self.device))
        model_plot.eval()

        with torch.no_grad():
            plot_preds_scaled_tensor = model_plot(X_plot_tensor)
            plot_preds_scaled = plot_preds_scaled_tensor.cpu().numpy().flatten()

        target_mean = self.target_scalers[cluster_id].mean_[0]
        target_std_dev = self.target_scalers[cluster_id].scale_[0]

        y_plot_original = (y_plot_scaled * target_std_dev) + target_mean
        plot_preds_original = (plot_preds_scaled * target_std_dev) + target_mean

        corresponding_indices_in_orig = cow_scaled_df.index[-len(y_plot_scaled):]
        dates_plot_direct = cow_data_orig.loc[corresponding_indices_in_orig, 'date'].values
        thi_plot_direct = cow_data_orig.loc[corresponding_indices_in_orig, 'THI'].values

        fig, ax1 = plt.subplots(figsize=(18, 7)) 

        ax1.set_xlabel("Data", fontsize=12)
        ax1.set_ylabel('Produzione Totale (kg)', color='blue', fontsize=12)
        ax1.plot(dates_plot_direct, y_plot_original, color='blue', marker='.', linestyle='-', markersize=4, label='Produzione reale')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.6) 
        
        ax1.plot(dates_plot_direct, plot_preds_original, color='orange', marker='x', linestyle='-', markersize=4, label='Produzione predetta')

        ax2 = ax1.twinx() 
        ax2.set_ylabel('THI', color='green', fontsize=12)
        ax2.plot(dates_plot_direct, thi_plot_direct, color='green', linestyle=':', label='THI')
        ax2.tick_params(axis='y', labelcolor='green')

        plt.title(f"Andamento Predetto - Mucca {cow_id}", fontsize=14)

        plt.tight_layout()
        ax1.legend()
        plt.show()

    def plot_cow(self, data: pd.DataFrame, cow_id: str, start_date: str = '2021-08-01', end_date: str = '2021-09-01'):
        if cow_id not in data['id_cow'].unique(): 
            raise Exception(f"ERRORE: Mucca {cow_id} non trovata nel DataFrame generale.")
        cow_data = data[data['id_cow'] == cow_id].copy().sort_values('date')
        cow_data = cow_data[cow_data['date'] >= pd.to_datetime(start_date)]
        cow_data = cow_data[cow_data['date'] <= pd.to_datetime(end_date)]
        
        fig, ax1 = plt.subplots(figsize=(18, 7)) 

        ax1.set_xlabel("Data", fontsize=12)
        ax1.set_ylabel('Produzione Totale (kg)', color='blue', fontsize=12)
        ax1.plot(cow_data['date'].to_numpy(), cow_data['tot_prod'].to_numpy(), color='blue', marker='.', linestyle='-', markersize=4, label='Produzione (kg)')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.6) 
        
        ax1.plot(cow_data['date'].to_numpy(), cow_data['cons'].to_numpy(), color='orange', marker='x', linestyle='--', markersize=4, label='Consumo (kg)')

        ax2 = ax1.twinx() 
        ax2.set_ylabel('THI', color='green', fontsize=12)
        ax2.plot(cow_data['date'].to_numpy(), cow_data['THI'].to_numpy(), color='green', linestyle=':', label='THI')
        ax2.tick_params(axis='y', labelcolor='green')

        plt.title(f"Andamento Storico - Mucca {cow_id}", fontsize=14)

        ax1.legend(fontsize=11)
        ax2.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
