import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

""" Preprocessing of the data """

def merge_with_cluster(data: pd.DataFrame, cluster_path):
    """ cluster_path must be numpy format"""
    labels = data["id_cow"].unique()
    clusters = np.load(cluster_path)

    results = pd.DataFrame({"id_cow": labels, "cluster": clusters})
    data = data.merge(results, on='id_cow', how='left')
    return data

def load_data(path: str, cluster: bool, cluster_path=None):
    df = pd.read_csv(path, parse_dates=["Data"], dayfirst=True)
    data = df.rename({"Data" : "date",
                    "Numero animale": "id_cow",
                    "Giorni di lattazione": "LP", # lactation period
                    "Numero lattazione": "LN", # lactation number
                    "Produzione totale": "tot_prod",
                    "Mungiture": "milkings",
                    "Mungiture incomplete": "incomplete_milkings",
                    "Media Durata mungitura": "avg_milk_duration",
                    "Mungiture con scalci": "milk_kicks",
                    "Stato riproduttivo": "RS", # reproductive status
                    "Consumato": "cons"}, axis=1)

    # Cleaning Data

    data["tot_prod"] = data["tot_prod"].astype(str).str.replace(",", ".").astype(float)
    data["cons"] = data["cons"].astype(str).str.replace(",", ".").astype(float)

    # Features Generation

    data["avg_milk_duration"].fillna(method="backfill", inplace=True)
    data["avg_milk_duration"] = pd.to_timedelta(["00:"+d for d in data.avg_milk_duration]).total_seconds()
    data['prod_avg'] = data.groupby('id_cow')['tot_prod'].transform(lambda x: x.mean())
    data['prod_var'] = data.groupby('id_cow')['tot_prod'].transform(lambda x: x.std())

    # Windowing

    window_size = 2
    data['avg_prod_avg'] = data.groupby('id_cow')['tot_prod'].transform(
        lambda x: x.rolling(window_size).mean().shift(1)
    )
    data['avg_prod_var'] = data.groupby('id_cow')['tot_prod'].transform(
        lambda x: x.rolling(window_size).var().shift(1)
    )
    data['thi_var'] = data.groupby('id_cow')['THI'].transform(
        lambda x: x.rolling(window_size).var().shift(1)
    )
    # Re-typing

    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by=['id_cow', 'date'])

    data['milk_diff'] = data.groupby('id_cow')['tot_prod'].diff()

    data['day_of_month'] = data['date'].dt.day
    data['month'] = data['date'].dt.month

    data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
    data['month_cos'] = np.cos(2 * np.pi * data['month']/12)

    data['prod_velocity'] = data.groupby('id_cow')['tot_prod'].diff()
    data['prod_acceleration'] = data.groupby('id_cow')['prod_velocity'].diff()
        
    if cluster:
        if cluster_path == None:
            raise Exception("cluster_path is required")
        else:
            data = merge_with_cluster(data, cluster_path)

    return data.dropna()