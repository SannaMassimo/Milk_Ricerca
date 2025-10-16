from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
import torch
import os

""" utils functions for training """

# Split the dataset in train and test set (test_size can be either the number of cows or the percentage of cows)
def split_cows_by_id(cluster_data, random_state=None, test_size=0.085):
    unique_cows = cluster_data['id_cow'].unique()

    # Calculate number of cow for the test
    if test_size < 1:
      n_test = max(1, int(len(unique_cows) * test_size))
    else:
      n_test = test_size

    if random_state is not None:
        np.random.seed(random_state)
    test_cows = np.random.choice(unique_cows, size=n_test, replace=False)

    train_data = cluster_data[~cluster_data['id_cow'].isin(test_cows)]
    test_data = cluster_data[cluster_data['id_cow'].isin(test_cows)]

    print(f"  Train: {train_data['id_cow'].nunique()} mucche, {len(train_data)} records")
    print(f"  Test:  {test_data['id_cow'].nunique()} mucche, {len(test_data)} records")

    return train_data, test_data

# Prepare sequences for LSTM
def prepare_sequences(df, feature_cols, target_col, sequence_length=8):
    sequences = []
    targets = []

    # Sort by cow and date
    df = df.sort_values(['id_cow', 'date'])
    grouped = df.groupby('id_cow')

    for cow_id, cow_df in grouped:
        # Check if the cow has enough data for at least one sequence + target
        if len(cow_df) > sequence_length:
            cow_features = cow_df[feature_cols].values.astype(np.float32)
            cow_target = cow_df[target_col].values.astype(np.float32)
            for i in range(len(cow_df) - sequence_length):
                sequences.append(cow_features[i:i+sequence_length])
                targets.append(cow_target[i+sequence_length])
        else:
            #print(f"Warning: Cow {cow_id} has only {len(cow_df)} records, which is not enough for a sequence of length {sequence_length}. Skipping this cow.")
            continue 
    
    if not sequences:
        print("Error: No sequences could be created from the provided data. Returning empty arrays.")
        return np.array([]), np.array([])

    return np.array(sequences), np.array(targets)


def prepareData(data, device, random_state, features, target_name, sequence_length, batch_size, test_size):
    train_data, test_data = split_cows_by_id(data, random_state, test_size=test_size)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(train_data[features])
    target_scaler.fit(train_data[[target_name]])

    train_features_scaled = feature_scaler.transform(train_data[features])
    test_features_scaled = feature_scaler.transform(test_data[features])
    train_target_scaled = target_scaler.transform(train_data[[target_name]]).flatten()
    test_target_scaled = target_scaler.transform(test_data[[target_name]]).flatten()

    train_scaled_df = pd.DataFrame(train_features_scaled, columns=features, index=train_data.index)
    train_scaled_df[target_name] = train_target_scaled
    test_scaled_df = pd.DataFrame(test_features_scaled, columns=features, index=test_data.index)
    test_scaled_df[target_name] = test_target_scaled

    train_scaled_df['id_cow'] = train_data['id_cow']
    train_scaled_df['date'] = train_data['date']
    test_scaled_df['id_cow'] = test_data['id_cow']
    test_scaled_df['date'] = test_data['date']

    train_scaled_df = train_scaled_df.dropna()
    test_scaled_df = test_scaled_df.dropna()

    print("Pre-calcolo di tutte le sequenze per il training set... (potrebbe richiedere tempo)")
    X_train, y_train = prepare_sequences(train_scaled_df, features, target_name, sequence_length)
    
    print("Pre-calcolo di tutte le sequenze per il test set...")
    X_test, y_test = prepare_sequences(test_scaled_df, features, target_name, sequence_length)
    print("Pre-calcolo completato.")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    use_gpu = device.type == 'cuda'
    
    num_workers = 0 
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_gpu)    
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)

    return train_loader, test_loader, feature_scaler, target_scaler

def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_permutation_importance(model, test_loader, feature_names, target_scaler, device):
    """
    Calcola la feature importance tramite permutazione sul set di test.
    Restituisce un DataFrame con l'importanza di ogni feature.
    """
    model.eval()

    # 1. Calcola la baseline loss (MSE) sul test set non modificato
    baseline_mse = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            outputs = model(batch_X)
            loss = nn.MSELoss()(outputs, batch_y)
            baseline_mse += loss.item() * batch_X.size(0)
    baseline_mse /= len(test_loader.dataset)
    print(f"Baseline Test MSE (scaled): {baseline_mse:.4f}\n")

    # Dizionario per salvare l'importanza
    importances = {}

    # 2. Itera su ogni feature
    for i, feature_name in enumerate(feature_names):
        print(f"Permuting feature: {feature_name}...")

        permuted_mse = 0.0

        # Copia i dati per non modificare l'originale
        X_test_original = test_loader.dataset.tensors[0].clone().cpu().numpy()
        y_test_original = test_loader.dataset.tensors[1].clone().cpu().numpy()

        # Permuta (mescola) solo la colonna della feature corrente
        np.random.shuffle(X_test_original[:, :, i])

        # Crea un nuovo DataLoader con i dati permutati
        permuted_dataset = TensorDataset(torch.tensor(X_test_original, dtype=torch.float32).to(device),
                                         torch.tensor(y_test_original, dtype=torch.float32).to(device))
        permuted_loader = DataLoader(permuted_dataset, batch_size=test_loader.batch_size)

        # 3. Calcola la loss con la feature permutata
        with torch.no_grad():
            for batch_X, batch_y in permuted_loader:
                batch_y = batch_y.unsqueeze(1)
                outputs = model(batch_X)
                loss = nn.MSELoss()(outputs, batch_y)
                permuted_mse += loss.item() * batch_X.size(0)
        permuted_mse /= len(permuted_loader.dataset)

        # 4. L'importanza è l'aumento della loss
        # Usiamo la differenza, ma si potrebbe usare anche il rapporto
        importance_score = permuted_mse - baseline_mse
        importances[feature_name] = importance_score

    # 5. Formatta e restituisci i risultati
    importance_df = pd.DataFrame.from_dict(importances, orient='index', columns=['Importance (Increase in MSE)'])
    importance_df = importance_df.sort_values(by='Importance (Increase in MSE)', ascending=False)

    return importance_df

# This class save the model if the validation loss decrease and stop the training if it doesn't decrease for a certain number of epochs
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        torch.save(self.best_model.state_dict(), self.path)
        self.val_loss_min = val_loss



from torch.utils.data import Dataset
class CowDataset(Dataset):
    def __init__(self, data: pd.DataFrame, features: list, target: str, sequence_length: int):
        self.sequence_length = sequence_length
        
        # Non creiamo le sequenze qui!
        # Raggruppiamo solo i dati per mucca per un accesso veloce.
        self.data_grouped = data.groupby('id_cow')
        self.cow_ids = list(self.data_grouped.groups.keys())
        
        # Calcoliamo gli "indici" di ogni possibile sequenza
        self.indices = []
        for cow_id in self.cow_ids:
            cow_data_len = len(self.data_grouped.get_group(cow_id))
            num_sequences = cow_data_len - sequence_length
            if num_sequences > 0:
                for i in range(num_sequences):
                    self.indices.append((cow_id, i))

        # Estraiamo i dati grezzi in array NumPy per velocità
        self.feature_data = {cow_id: group[features].values for cow_id, group in self.data_grouped}
        self.target_data = {cow_id: group[target].values for cow_id, group in self.data_grouped}

    def __len__(self):
        # La lunghezza del dataset è il numero totale di sequenze che possiamo creare
        return len(self.indices)

    def __getitem__(self, idx):
        # Questo metodo viene chiamato da un worker del DataLoader
        # per ottenere UN singolo campione (una sequenza e il suo target)
        
        cow_id, start_idx = self.indices[idx]
        
        end_idx = start_idx + self.sequence_length
        
        # Estrae la fetta di dati per la sequenza e il target
        sequence = self.feature_data[cow_id][start_idx:end_idx]
        target = self.target_data[cow_id][end_idx]
        
        # Converte in tensori al momento
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)