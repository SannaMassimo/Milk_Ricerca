import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Importa le utility dal tuo progetto
from utilsTraining import split_cows_by_id, prepare_sequences

class BaselineRunner:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.features = config['features']
        self.target_name = config['training']['target_name']
        self.seq_length = config['training']['sequence_length']
        self.horizon = config['training'].get('prediction_horizon', 1)
        self.random_state = config['training']['random_state']
        self.results = {}
        
        # Check impostazione corretta
        if self.horizon != 1:
            print(f"INFO: Baseline in esecuzione con horizon={self.horizon}.")

    def _prepare_ml_data(self):
        """
        Converte le sequenze 3D (N, T, F) in 2D (N, T*F) per i modelli classici.
        Gestisce correttamente l'appiattimento del target se horizon=1.
        """
        print("Preparazione dati (creazione sequenze)...")
        train_data, test_data = split_cows_by_id(self.data, 
                                                 random_state=self.random_state, 
                                                 test_size=self.config['training']['test_size'])
        
        # prepare_sequences restituisce y come (N, horizon)
        X_train_3d, y_train = prepare_sequences(train_data, self.features, self.target_name, self.seq_length, self.horizon)
        X_test_3d, y_test = prepare_sequences(test_data, self.features, self.target_name, self.seq_length, self.horizon)

        # Flatten inputs: (N, 10, 12) -> (N, 120)
        X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
        X_test_2d = X_test_3d.reshape(X_test_3d.shape[0], -1)

        # Se horizon è 1, appiattiamo y per evitare warning di sklearn: (N, 1) -> (N,)
        if self.horizon == 1:
            y_train = y_train.ravel()
            y_test = y_test.ravel()

        print(f" -> Dati Pronti. Train: {X_train_2d.shape}, Test: {X_test_2d.shape}")
        return X_train_2d, y_train, X_test_2d, y_test

    def run_random_forest(self):
        print("\n" + "="*40)
        print(f"Running Random Forest Baseline ({self.horizon} Day Prediction)")
        print("="*40)
        
        X_train, y_train, X_test, y_test = self._prepare_ml_data()

        # Configurazione Random Forest
        n_estimators_total = 100
        rf = RandomForestRegressor(
            n_estimators=0,      # Partiamo da 0 per usare warm_start
            warm_start=True,     # Permette di aggiungere alberi incrementalmente
            random_state=self.random_state,
            n_jobs=-1            # Usa tutti i core della CPU
        )

        print(f"Training Random Forest ({n_estimators_total} alberi)...")
        
        # TRUCCO PER LA PROGRESS BAR:
        chunk_size = 10
        with tqdm(total=n_estimators_total, desc="Training Trees", unit="tree") as pbar:
            for i in range(0, n_estimators_total, chunk_size):
                rf.n_estimators += chunk_size
                if rf.n_estimators > n_estimators_total:
                    rf.n_estimators = n_estimators_total
                
                rf.fit(X_train, y_train)
                pbar.update(chunk_size)

        print("Predicting...")
        preds = rf.predict(X_test)
        
        self._compute_metrics("Random Forest", y_test, preds)
        return rf

    def run_xgboost(self):
        print("\n" + "="*40)
        print(f"Running XGBoost Baseline ({self.horizon} Day Prediction)")
        print("="*40)
        
        X_train, y_train, X_test, y_test = self._prepare_ml_data()

        # XGBoost configuration
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=-1
        )

        print("Training XGBoost...")
        xgb_model.fit(X_train, y_train)
        
        print("Predicting...")
        preds = xgb_model.predict(X_test)
        
        self._compute_metrics("XGBoost", y_test, preds)
        return xgb_model

    def run_arima_baseline(self):
        print("\n" + "="*40)
        print("Running ARIMA Baseline (Univariate)")
        print("="*40)
        
        # Carichiamo solo i dati di test perché ARIMA si allena sulla storia locale
        _, test_data = split_cows_by_id(self.data, 
                                        random_state=self.random_state, 
                                        test_size=self.config['training']['test_size'])
        
        all_preds = []
        all_targets = []
        
        test_cows = test_data['id_cow'].unique()
        print(f"Valutazione ARIMA su {len(test_cows)} mucche...")
        print("NOTA: ARIMA è lento. Sto processando un sottoinsieme casuale di finestre temporali per velocità.")

        # Barra di avanzamento sulle mucche
        for cow_id in tqdm(test_cows, desc="Processing Cows"):
            cow_df = test_data[test_data['id_cow'] == cow_id].sort_values('date')
            series = cow_df[self.target_name].values
            
            # Indici possibili dove possiamo fare previsione
            # (Servono seq_length giorni di storia prima)
            valid_indices = range(0, len(series) - self.seq_length - self.horizon + 1)
            
            # Se la mucca ha pochi dati, saltiamo o ne prendiamo meno
            indices = list(valid_indices)
            
            # SUBSAMPLING: Per non farci mettere 10 ore, prendiamo max 30 predizioni casuali per mucca
            if len(indices) > 30: 
                indices = np.random.choice(indices, 30, replace=False)

            for i in indices:
                # Definiamo la finestra di storia (training set locale per ARIMA)
                # Usiamo fino a 60 giorni di storia passata se disponibile per fitting veloce
                history_start = max(0, i + self.seq_length - 60)
                history_end = i + self.seq_length
                
                history = series[history_start : history_end]
                
                # Gestione target (scalare o vettore)
                if self.horizon == 1:
                    target_val = series[history_end]
                else:
                    target_val = series[history_end : history_end + self.horizon]

                if len(history) < 10: continue # Troppo pochi dati per ARIMA

                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        # ARIMA(5,1,0) è standard veloce e robusto per trend
                        model = ARIMA(history, order=(5,1,0)) 
                        model_fit = model.fit()
                        
                        # Previsione N step
                        forecast = model_fit.forecast(steps=self.horizon)
                        
                        if self.horizon == 1:
                            forecast = forecast[0]

                        all_preds.append(forecast)
                        all_targets.append(target_val)
                except Exception:
                    continue

        if len(all_preds) > 0:
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            self._compute_metrics("ARIMA", all_targets, all_preds)
        else:
            print("ARIMA fallito (nessuna predizione generata).")

    def _compute_metrics(self, name, y_true, y_pred):
        # Assicuriamoci che le dimensioni combacino appiattendo tutto
        # Questo permette di calcolare un MSE globale anche se horizon > 1
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Filtro NaN per sicurezza
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            print(f"Nessun dato valido per metriche {name}")
            return

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"\nRisultati per {name}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
        
        self.results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    def print_summary(self):
        print("\n" + "="*40)
        print("RIEPILOGO COMPARATIVO BASELINE")
        print("="*40)
        if not self.results:
            print("Nessun risultato disponibile.")
            return
            
        df = pd.DataFrame(self.results).T
        # Ordiniamo per RMSE crescente (migliore in alto)
        if 'RMSE' in df.columns:
            df = df.sort_values('RMSE')
            
        print(df)
        print("="*40)