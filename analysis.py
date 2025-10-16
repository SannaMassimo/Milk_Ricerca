import torch
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from training import LSTMModel
from utilsTraining import prepare_sequences, run_permutation_importance

class ModelAnalyzer:
    def __init__(self, model_path: str, config: dict):
        """
        Inizializza l'analizzatore con il percorso a un modello allenato e la sua config.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"La directory del modello non esiste: {model_path}")

        self.model_path = model_path
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Carica modelli e scaler
        self.models = {}
        self.feature_scalers = {}
        self.target_scalers = {}
        self._load_models_and_scalers()

    def _load_models_and_scalers(self):
        """Metodo privato per caricare tutti i modelli e gli scaler per ogni cluster."""
        num_clusters = self.config['clustering']['n_clusters']
        features = self.config['features']
        model_params = self.config['training']['model_params']

        for i in range(num_clusters):
            # Carica il modello
            model_file = os.path.join(self.model_path, f'lstm_model_cluster_{i}.pt')
            model = LSTMModel(input_size=len(features), **model_params)
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[i] = model
            
            # Carica gli scaler
            feature_scaler_path = os.path.join(self.model_path, f'feature_scaler_cluster_{i}.pkl')
            target_scaler_path = os.path.join(self.model_path, f'target_scaler_cluster_{i}.pkl')
            self.feature_scalers[i] = joblib.load(feature_scaler_path)
            self.target_scalers[i] = joblib.load(target_scaler_path)
        print(f"Caricati {len(self.models)} modelli e i relativi scaler da {self.model_path}")

    def analyze_performance(self, test_data: pd.DataFrame):
        """
        Esegue un'analisi completa delle performance su un DataFrame di dati di test.
        """
        all_preds, all_targets, all_residuals = [], [], []
        cluster_metrics = {}

        for cluster_id, model in self.models.items():
            print(f"\n--- Analisi Cluster: {cluster_id} ---")
            
            cluster_test_data = test_data[test_data['cluster'] == cluster_id].copy()
            if cluster_test_data.empty:
                print(f"Nessun dato di test per il cluster {cluster_id}. Salto.")
                continue

            # Prepara i dati per questo cluster
            features = self.config['features']
            target_name = self.config['training']['target_name']
            
            test_features_scaled = self.feature_scalers[cluster_id].transform(cluster_test_data[features])
            test_target_scaled = self.target_scalers[cluster_id].transform(cluster_test_data[[target_name]]).flatten()

            test_scaled_df = pd.DataFrame(test_features_scaled, columns=features)
            test_scaled_df[target_name] = test_target_scaled
            test_scaled_df['id_cow'] = cluster_test_data['id_cow'].values

            X_test, y_test_scaled = prepare_sequences(test_scaled_df, features, target_name, self.config['training']['sequence_length'])

            if X_test.shape[0] == 0:
                print(f"Non è stato possibile creare sequenze per il test set del cluster {cluster_id}. Salto.")
                continue

            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_scaled, dtype=torch.float32))
            test_loader = DataLoader(test_dataset, batch_size=self.config['training']['hyperparameters']['batch_size'])

            # Ottieni predizioni
            preds_scaled, targets_scaled = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    outputs = model(batch_X)
                    preds_scaled.extend(outputs.cpu().numpy().flatten())
                    targets_scaled.extend(batch_y.cpu().numpy().flatten())

            # De-scala per metriche interpretabili
            preds_original = self.target_scalers[cluster_id].inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
            targets_original = self.target_scalers[cluster_id].inverse_transform(np.array(targets_scaled).reshape(-1, 1)).flatten()
            
            # Calcola metriche
            metrics = self._calculate_metrics(targets_original, preds_original)
            cluster_metrics[cluster_id] = metrics
            
            print(f"  MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.2f}%")
            
            # Aggiungi a liste aggregate
            all_preds.extend(preds_original)
            all_targets.extend(targets_original)
            all_residuals.extend(targets_original - preds_original)

        print("\n--- Performance Complessiva (Aggregata su tutti i cluster) ---")
        overall_metrics = self._calculate_metrics(np.array(all_targets), np.array(all_preds))
        print(f"  MSE: {overall_metrics['mse']:.4f}, RMSE: {overall_metrics['rmse']:.4f}, MAE: {overall_metrics['mae']:.4f}, R2: {overall_metrics['r2']:.4f}, MAPE: {overall_metrics['mape']:.2f}%")
        
        self.plot_residuals(np.array(all_residuals), np.array(all_preds))
        
        return cluster_metrics, overall_metrics

    def analyze_feature_importance(self, test_data: pd.DataFrame):
        """
        Esegue l'analisi di Permutation Feature Importance per ogni cluster.
        """
        for cluster_id, model in self.models.items():
            print(f"\n--- Analisi Feature Importance per Cluster: {cluster_id} ---")
            
            # (La logica per preparare il test_loader è identica a quella sopra)
            # Potrebbe essere rifattorizzata in un metodo privato per non duplicare il codice
            # Per semplicità, la replico qui
            cluster_test_data = test_data[test_data['cluster'] == cluster_id].copy()
            if cluster_test_data.empty: continue
            
            features = self.config['features']
            target_name = self.config['training']['target_name']
            
            test_features_scaled = self.feature_scalers[cluster_id].transform(cluster_test_data[features])
            test_target_scaled = self.target_scalers[cluster_id].transform(cluster_test_data[[target_name]]).flatten()

            test_scaled_df = pd.DataFrame(test_features_scaled, columns=features)
            test_scaled_df[target_name] = test_target_scaled
            test_scaled_df['id_cow'] = cluster_test_data['id_cow'].values

            X_test, y_test_scaled = prepare_sequences(test_scaled_df, features, target_name, self.config['training']['sequence_length'])
            
            if X_test.shape[0] == 0: continue
            
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_scaled, dtype=torch.float32))
            test_loader = DataLoader(test_dataset, batch_size=self.config['training']['hyperparameters']['batch_size'])

            # Esegui l'analisi
            importance_df = run_permutation_importance(model, test_loader, features, self.target_scalers[cluster_id], self.device)
            print("Risultati Feature Importance:")
            print(importance_df)

            # Plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importance_df['Importance (Increase in MSE)'], y=importance_df.index)
            plt.title(f'Permutation Feature Importance per Cluster {cluster_id}')
            plt.xlabel('Aumento della MSE (su dati scalati)')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()

    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        """Metodo statico helper per calcolare un dizionario di metriche."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        valid_indices = y_true != 0
        mape = np.mean(np.abs((y_true[valid_indices] - y_pred[valid_indices]) / y_true[valid_indices])) * 100 if np.sum(valid_indices) > 0 else float('nan')

        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

    @staticmethod
    def plot_residuals(residuals, predictions):
        """Metodo statico per plottare i residui."""
        plt.figure(figsize=(12, 6))
        plt.suptitle("Analisi dei Residui Aggregata")
        
        plt.subplot(1, 2, 1)
        sns.histplot(residuals, kde=True)
        plt.title('Istogramma Residui')
        plt.xlabel('Residuo (Reale - Predetto)')
        
        plt.subplot(1, 2, 2)
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residui vs. Predizioni')
        plt.xlabel('Produzione Predetta (kg)')
        plt.ylabel('Residuo (kg)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()