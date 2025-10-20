# In analysis.py

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
        
        self.models = {}
        self.feature_scalers = {}
        self.target_scalers = {}
        self._load_models_and_scalers()

    def run_full_analysis(self, test_data: pd.DataFrame):
        _, lstm_metrics = self._analyze_performance(test_data)
        prev_day_baseline_metrics = self._analyze_prev_day_baseline(test_data)
        
        rolling_mean_baseline_metrics = self._analyze_rolling_mean_baseline(test_data, window_size=5)
        
        self._print_comparison_table(lstm_metrics, prev_day_baseline_metrics, rolling_mean_baseline_metrics)
        
        self._analyze_feature_importance(test_data)

    def _load_models_and_scalers(self):
        num_clusters = self.config['clustering']['n_clusters']
        features = self.config['features']
        model_params = self.config['training']['model_params']
        model_exists = any(f.startswith('lstm_model_cluster_') for f in os.listdir(self.model_path))
        if not model_exists:
            print(f"Attenzione: Nessun modello trovato in {self.model_path}. L'analisi non può continuare.")
            return
        for i in range(num_clusters):
            model_file = os.path.join(self.model_path, f'lstm_model_cluster_{i}.pt')
            if not os.path.exists(model_file): continue
            model = LSTMModel(input_size=len(features), **model_params)
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[i] = model
            feature_scaler_path = os.path.join(self.model_path, f'feature_scaler_cluster_{i}.pkl')
            target_scaler_path = os.path.join(self.model_path, f'target_scaler_cluster_{i}.pkl')
            self.feature_scalers[i] = joblib.load(feature_scaler_path)
            self.target_scalers[i] = joblib.load(target_scaler_path)
        if self.models: print(f"Caricati {len(self.models)} modelli e i relativi scaler da {self.model_path}")
        else: print("Errore critico: non è stato caricato nessun modello.")

    def _analyze_performance(self, test_data: pd.DataFrame):
        all_preds, all_targets, all_residuals = [], [], []
        print("\n--- Analisi Performance Modello LSTM (dettagli per cluster) ---")
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
            X_test, y_test_scaled = prepare_sequences(test_scaled_df, features, target_name, self.config['training']['sequence_length'])
            if X_test.shape[0] == 0: continue
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_scaled, dtype=torch.float32))
            test_loader = DataLoader(test_dataset, batch_size=self.config['training']['hyperparameters']['batch_size'])
            preds_scaled, targets_scaled = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    outputs = model(batch_X)
                    preds_scaled.extend(outputs.cpu().numpy().flatten())
                    targets_scaled.extend(batch_y.cpu().numpy().flatten())
            preds_original = self.target_scalers[cluster_id].inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
            targets_original = self.target_scalers[cluster_id].inverse_transform(np.array(targets_scaled).reshape(-1, 1)).flatten()
            all_preds.extend(preds_original); all_targets.extend(targets_original); all_residuals.extend(targets_original - preds_original)
        self._plot_residuals(np.array(all_residuals), np.array(all_preds))
        overall_metrics = self._calculate_metrics(np.array(all_targets), np.array(all_preds))
        return None, overall_metrics

    def _analyze_prev_day_baseline(self, test_data: pd.DataFrame):
        all_baseline_preds, all_baseline_targets = [], []
        
        baseline_df = test_data[['id_cow', 'date', 'tot_prod']].copy()
        baseline_df.sort_values(['id_cow', 'date'], inplace=True)
        baseline_df['prediction'] = baseline_df.groupby('id_cow')['tot_prod'].shift(1)
        baseline_df.dropna(inplace=True)
        
        all_baseline_preds.extend(baseline_df['prediction'].values)
        all_baseline_targets.extend(baseline_df['tot_prod'].values)

        return self._calculate_metrics(np.array(all_baseline_targets), np.array(all_baseline_preds))

    def _analyze_rolling_mean_baseline(self, test_data: pd.DataFrame, window_size: int):
        all_baseline_preds, all_baseline_targets = [], []
        
        baseline_df = test_data[['id_cow', 'date', 'tot_prod']].copy()
        baseline_df.sort_values(['id_cow', 'date'], inplace=True)
        
        # Calcola la media mobile sui 'window_size' giorni precedenti per ogni mucca
        baseline_df['prediction'] = baseline_df.groupby('id_cow')['tot_prod'].transform(
            lambda x: x.rolling(window=window_size).mean().shift(1)
        )
        
        baseline_df.dropna(inplace=True)
        
        all_baseline_preds.extend(baseline_df['prediction'].values)
        all_baseline_targets.extend(baseline_df['tot_prod'].values)

        return self._calculate_metrics(np.array(all_baseline_targets), np.array(all_baseline_preds))
        
    def _print_comparison_table(self, lstm_metrics, prev_day_metrics, rolling_mean_metrics):
        if not all([lstm_metrics, prev_day_metrics, rolling_mean_metrics]):
            print("\nImpossibile generare la tabella di confronto: metriche mancanti.")
            return

        metric_names = {'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R-squared', 'mape': 'MAPE (%)'}
        
        comparison_df = pd.DataFrame({
            'LSTM Model': pd.Series({metric_names[k]: v for k, v in lstm_metrics.items()}),
            'Baseline (Prev Day)': pd.Series({metric_names[k]: v for k, v in prev_day_metrics.items()}),
            'Baseline (5-Day Avg)': pd.Series({metric_names[k]: v for k, v in rolling_mean_metrics.items()})
        })

        print("\n\n" + "="*80)
        print("--- TABELLA DI CONFRONTO PERFORMANCE COMPLESSIVA ---")
        print("="*80)
        print(comparison_df.to_string(
            formatters={
                'LSTM Model': '{:,.3f}'.format,
                'Baseline (Prev Day)': '{:,.3f}'.format,
                'Baseline (5-Day Avg)': '{:,.3f}'.format,
            }
        ))
        print("="*80 + "\n")

    def _analyze_feature_importance(self, test_data: pd.DataFrame):
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
            
            X_test, y_test_scaled = prepare_sequences(test_scaled_df, features, target_name, self.config['training']['sequence_length'])
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
        
        print("\n--- Analisi Feature Importance Complessiva (Media Ponderata) ---")
        print(overall_importance.to_string())
        plt.figure(figsize=(12, 8)); sns.barplot(x=overall_importance.values, y=overall_importance.index, palette='viridis')
        plt.title('Permutation Feature Importance Complessiva (Aggregata)', fontsize=16); plt.xlabel('Aumento della MSE (su dati scalati)', fontsize=12); plt.ylabel('Feature', fontsize=12)
        plt.tight_layout(); plt.show()

    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        if np.isnan(y_true).any() or np.isnan(y_pred).any(): return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan}
        mse = mean_squared_error(y_true, y_pred)
        return {'mse': mse, 'rmse': np.sqrt(mse), 'mae': mean_absolute_error(y_true, y_pred), 'r2': r2_score(y_true, y_pred), 'mape': np.mean(np.abs((y_true - y_pred) / y_true)[y_true != 0]) * 100}

    @staticmethod
    def _plot_residuals(residuals, predictions):
        if len(residuals) == 0: return
        plt.figure(figsize=(12, 6)); plt.suptitle("Analisi dei Residui Aggregata"); plt.subplot(1, 2, 1); sns.histplot(residuals, kde=True)
        plt.title('Istogramma Residui'); plt.xlabel('Residuo (Reale - Predetto)'); plt.subplot(1, 2, 2); plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--'); plt.title('Residui vs. Predizioni'); plt.xlabel('Produzione Predetta (kg)'); plt.ylabel('Residuo (kg)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()