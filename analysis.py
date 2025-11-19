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
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"La directory del modello non esiste: {model_path}")

        self.model_path = model_path
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prediction_horizon = config['training'].get('prediction_horizon', 1)
        
        self.models = {}
        self.feature_scalers = {}
        self.target_scalers = {}
        self._load_models_and_scalers()

    def run_full_analysis(self, test_data: pd.DataFrame):
        _, lstm_metrics = self._analyze_performance(test_data)
        prev_day_metrics = self._analyze_naive_baseline(test_data, method='last_value')
        rolling_mean_metrics = self._analyze_naive_baseline(test_data, method='rolling_mean', window=5)
        
        self._print_comparison_table(lstm_metrics, prev_day_metrics, rolling_mean_metrics)
        self._analyze_feature_importance(test_data)

    def _load_models_and_scalers(self):
        num_clusters = self.config['clustering']['n_clusters']
        features = self.config['features']
        model_params = self.config['training']['model_params']
        
        model_exists = any(f.startswith('lstm_model_cluster_') for f in os.listdir(self.model_path))
        if not model_exists:
            print(f"Attenzione: Nessun modello trovato in {self.model_path}.")
            return
            
        for i in range(num_clusters):
            model_file = os.path.join(self.model_path, f'lstm_model_cluster_{i}.pt')
            if not os.path.exists(model_file): continue
            
            # Passiamo output_size
            model = LSTMModel(input_size=len(features), **model_params, output_size=self.prediction_horizon)
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[i] = model
            
            feature_scaler_path = os.path.join(self.model_path, f'feature_scaler_cluster_{i}.pkl')
            target_scaler_path = os.path.join(self.model_path, f'target_scaler_cluster_{i}.pkl')
            self.feature_scalers[i] = joblib.load(feature_scaler_path)
            self.target_scalers[i] = joblib.load(target_scaler_path)
            
        print(f"Caricati {len(self.models)} modelli. Horizon: {self.prediction_horizon} giorni.")

    def _analyze_performance(self, test_data: pd.DataFrame):
        all_preds, all_targets = [], []
        print("\n--- Analisi Performance LSTM (Multi-Step) ---")
        
        for cluster_id, model in self.models.items():
            cluster_test_data = test_data[test_data['cluster'] == cluster_id].copy()
            if cluster_test_data.empty: continue
            features = self.config['features']
            target_name = self.config['training']['target_name']
            
            test_features_scaled = self.feature_scalers[cluster_id].transform(cluster_test_data[features])
            test_target_scaled = self.target_scalers[cluster_id].transform(cluster_test_data[[target_name]]).flatten()
            
            test_scaled_df = pd.DataFrame(test_features_scaled, columns=features, index=cluster_test_data.index)
            test_scaled_df[target_name] = test_target_scaled
            test_scaled_df['id_cow'] = cluster_test_data['id_cow']
            test_scaled_df['date'] = cluster_test_data['date']
            test_scaled_df = test_scaled_df.dropna()
            
            X_test, y_test_scaled = prepare_sequences(test_scaled_df, features, target_name, 
                                                      self.config['training']['sequence_length'],
                                                      self.prediction_horizon)
            
            if X_test.shape[0] == 0: continue
            
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_scaled, dtype=torch.float32))
            test_loader = DataLoader(test_dataset, batch_size=self.config['training']['hyperparameters']['batch_size'])
            
            cluster_preds_scaled, cluster_targets_scaled = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    outputs = model(batch_X) # Output (batch, 7)
                    cluster_preds_scaled.append(outputs.cpu().numpy())
                    cluster_targets_scaled.append(batch_y.cpu().numpy())
            
            if not cluster_preds_scaled: continue

            # Concatenazione batch
            cluster_preds_scaled = np.concatenate(cluster_preds_scaled, axis=0)   # (N, 7)
            cluster_targets_scaled = np.concatenate(cluster_targets_scaled, axis=0) # (N, 7)

            # Inverse transform manuale (Scaler è 1D, dati sono 2D)
            mean = self.target_scalers[cluster_id].mean_[0]
            std = self.target_scalers[cluster_id].scale_[0]
            
            preds_original = (cluster_preds_scaled * std) + mean
            targets_original = (cluster_targets_scaled * std) + mean
            
            all_preds.append(preds_original)
            all_targets.append(targets_original)

        if not all_preds: return None, None
        
        # Concatena tutti i cluster
        all_preds_np = np.concatenate(all_preds, axis=0) # (Tot_Samples, 7)
        all_targets_np = np.concatenate(all_targets, axis=0) # (Tot_Samples, 7)
        
        # Per l'istogramma dei residui, appiattiamo tutto (trattiamo ogni predizione giornaliera come un campione)
        residuals_flat = (all_targets_np - all_preds_np).flatten()
        self._plot_residuals(residuals_flat, all_preds_np.flatten())
        
        overall_metrics = self._calculate_metrics(all_targets_np, all_preds_np)
        return None, overall_metrics

    def _analyze_naive_baseline(self, test_data: pd.DataFrame, method='last_value', window=5):
        """
        Baseline Multi-Step:
        - last_value: ripete l'ultimo valore noto per 7 giorni.
        - rolling_mean: calcola la media degli ultimi X giorni e la proietta per 7 giorni.
        """
        all_preds, all_targets = [], []
        features = self.config['features']
        target_name = self.config['training']['target_name']
        
        # Utilizziamo prepare_sequences per ottenere gli allineamenti corretti di X e y
        # Ma non usiamo X per l'LSTM, usiamo l'ultimo valore di X per la baseline
        
        test_data = test_data.sort_values(['id_cow', 'date'])
        
        # Non scaliamo i dati qui, lavoriamo sui dati originali per la baseline
        X_struct, y_struct = prepare_sequences(test_data, features, target_name, 
                                               self.config['training']['sequence_length'],
                                               self.prediction_horizon)
        
        # X_struct è (N, seq, features). La colonna target (tot_prod) deve essere tra le features o recuperata
        # Per semplicità, recuperiamo l'ultimo valore della sequenza di 'tot_prod' da X_struct
        # Cerchiamo l'indice di 'tot_prod' nelle features
        try:
            target_idx = features.index(target_name)
        except ValueError:
            # Se tot_prod non è nelle features, non possiamo fare baseline facilmente su X_struct
            # Assumiamo che tot_prod sia tra le features come nel config originale
            print("Warning: Target non presente nelle features per la baseline.")
            return {}

        last_values = X_struct[:, -1, target_idx] # (N,) Ultimo valore della sequenza
        
        if method == 'rolling_mean':
            # Media degli ultimi 'window' step della sequenza
            # Prendiamo gli ultimi 'window' step
            last_window = X_struct[:, -window:, target_idx] # (N, window)
            baseline_val = np.mean(last_window, axis=1) # (N,)
        else:
            # last_value
            baseline_val = last_values
            
        # Proiettiamo per l'horizon (N, horizon)
        baseline_preds = np.tile(baseline_val.reshape(-1, 1), (1, self.prediction_horizon))
        
        return self._calculate_metrics(y_struct, baseline_preds)

    def _print_comparison_table(self, lstm_metrics, prev_day_metrics, rolling_mean_metrics):
        if not all([lstm_metrics, prev_day_metrics, rolling_mean_metrics]):
            print("\nMetriche mancanti.")
            return

        metric_names = {'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R2'}
        
        df = pd.DataFrame({
            'LSTM (7 Days)': pd.Series({k: v for k, v in lstm_metrics.items() if k in metric_names}),
            'Repeat Last Day': pd.Series({k: v for k, v in prev_day_metrics.items() if k in metric_names}),
            'Repeat Rolling Avg': pd.Series({k: v for k, v in rolling_mean_metrics.items() if k in metric_names})
        })
        
        print("\n" + "="*60)
        print(f"--- CONFRONTO PERFORMANCE (Media sui {self.prediction_horizon} giorni) ---")
        print("="*60)
        print(df)
        print("="*60 + "\n")

    def _analyze_feature_importance(self, test_data: pd.DataFrame):
        # Identica logica, permutation importance funziona anche con output multipli (valuta aumento MSE totale)
        all_importance_dfs = []
        for cluster_id, model in self.models.items():
            cluster_test_data = test_data[test_data['cluster'] == cluster_id].copy()
            if cluster_test_data.empty: continue
            
            features = self.config['features']
            target_name = self.config['training']['target_name']
            
            test_features_scaled = self.feature_scalers[cluster_id].transform(cluster_test_data[features])
            test_target_scaled = self.target_scalers[cluster_id].transform(cluster_test_data[[target_name]]).flatten()
            
            test_scaled_df = pd.DataFrame(test_features_scaled, columns=features, index=cluster_test_data.index)
            test_scaled_df[target_name] = test_target_scaled
            test_scaled_df['id_cow'] = cluster_test_data['id_cow']
            test_scaled_df['date'] = cluster_test_data['date']
            test_scaled_df = test_scaled_df.dropna()
            
            X_test, y_test_scaled = prepare_sequences(test_scaled_df, features, target_name, 
                                                      self.config['training']['sequence_length'],
                                                      self.prediction_horizon)
            if X_test.shape[0] == 0: continue
            
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_scaled, dtype=torch.float32))
            test_loader = DataLoader(test_dataset, batch_size=self.config['training']['hyperparameters']['batch_size'])
            
            importance_df = run_permutation_importance(model, test_loader, features, self.target_scalers[cluster_id], self.device)
            importance_df['weight'] = len(test_loader.dataset)
            all_importance_dfs.append(importance_df.reset_index().rename(columns={'index': 'feature'}))

        if not all_importance_dfs: return
        
        combined_df = pd.concat(all_importance_dfs)
        combined_df['weighted_importance'] = combined_df['Importance (Increase in MSE)'] * combined_df['weight']
        grouped = combined_df.groupby('feature')
        overall_importance = (grouped['weighted_importance'].sum() / grouped['weight'].sum()).sort_values(ascending=False)
        
        print("\n--- Feature Importance Complessiva (Aggregata) ---")
        print(overall_importance.to_string())

    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        # y_true e y_pred sono matrici (N, 7)
        # MSE, MAE calcolano la media su tutti gli elementi appiattiti (default behavior spesso desiderato)
        # oppure 'uniform_average' su multioutput
        if np.isnan(y_true).any() or np.isnan(y_pred).any(): return {}
        
        mse = mean_squared_error(y_true, y_pred) # Average over all elements
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred) # R2 score aggregate is tricky, but sklearn handles it
        
        return {'mse': mse, 'rmse': np.sqrt(mse), 'mae': mae, 'r2': r2}

    @staticmethod
    def _plot_residuals(residuals, predictions):
        if len(residuals) == 0: return
        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Analisi dei Residui (Tutti i giorni predetti)")
        plt.subplot(1, 2, 1); sns.histplot(residuals, kde=True, bins=50)
        plt.title('Distribuzione Errori'); plt.xlabel('Residuo (kg)')
        plt.subplot(1, 2, 2); plt.scatter(predictions, residuals, alpha=0.1, s=1)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residui vs. Predizioni'); plt.xlabel('Predizione (kg)'); plt.ylabel('Residuo (kg)')
        plt.tight_layout()
        plt.show()