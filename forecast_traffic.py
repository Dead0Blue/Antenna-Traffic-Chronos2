import pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

def parse_french_date(date_str):
    parts = str(date_str).split()
    if len(parts) >= 4:
        day = parts[1]
        month_fr = parts[2].lower()
        year = parts[3]
        months = {
            'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04',
            'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08',
            'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
        }
        month = months.get(month_fr, '01')
        return f"{year}-{month}-{day.zfill(2)}"
    return None

def load_data(file_path):
    with open(file_path, 'r', encoding='latin1') as f:
        lines = f.readlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if 'secteur' in line.lower() and 'tstamp' in line.lower():
            header_idx = i
            break
    df = pd.read_csv(file_path, sep=';', encoding='latin1', skiprows=header_idx)
    df = df.dropna(axis=1, how='all')
    df.columns = [c.strip() for c in df.columns]
    df = df[df['secteur'] != 'secteur']
    df['date'] = df['tstamp'].apply(parse_french_date)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'trafic_mbps', 'secteur'])
    df['trafic_mbps'] = pd.to_numeric(df['trafic_mbps'], errors='coerce')
    df = df.dropna(subset=['trafic_mbps'])
    df = df.sort_values(['secteur', 'date'])
    return df

def main():
    csv_path = r'c:\Users\PC\Desktop\Supélec\S8\Projet S8\document 1\histo_trafic.csv'
    print("Loading data...")
    df = load_data(csv_path)
    
    sectors = df['secteur'].unique()
    print(f"Found {len(sectors)} sectors.")
    
    clean_context_list = []
    ground_truth = {}
    
    print("Cleaning sectors and preparing context...")
    for sector in sectors:
        sector_df = df[df['secteur'] == sector].drop_duplicates('date').sort_values('date')
        if len(sector_df) < 10:
            continue
            
        # Regularize frequency to 7 days
        sector_df = sector_df.set_index('date').asfreq('7D')
        sector_df.index.name = 'date'
        sector_df['secteur'] = sector
        # Fill gaps
        sector_df['trafic_mbps'] = sector_df['trafic_mbps'].interpolate().ffill().bfill()
        
        # Split into context (all except last) and ground truth (last)
        clean_context_list.append(sector_df.iloc[:-1].reset_index())
        ground_truth[sector] = sector_df.iloc[-1]['trafic_mbps']
    
    context_df = pd.concat(clean_context_list)
    
    print("Initialising Chronos-2 pipeline...")
    pipeline = Chronos2Pipeline.from_pretrained(
        "autogluon/chronos-2-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    
    print("Predicting with predict_df...")
    forecast_df = pipeline.predict_df(
        context_df,
        prediction_length=1,
        id_column="secteur",
        timestamp_column="date",
        target="trafic_mbps",
        quantile_levels=[0.5]
    )
    
    print("Calculating RMSE...")
    results = []
    for _, row in forecast_df.iterrows():
        sector = row['secteur']
        if sector in ground_truth:
            results.append({
                'secteur': sector,
                'actual': ground_truth[sector],
                'predicted': row['0.5']
            })
    
    results_df = pd.DataFrame(results)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(results_df['actual'], results_df['predicted']))
    print(f"\nRoot Mean Square Error (RMSE): {rmse:.4f}")
    
    # Save results
    results_df.to_csv("prediction_results.csv", index=False)
    print("Results saved to prediction_results.csv")
    
    # Sample results
    print("\nSample Results (First 5):")
    print(results_df.head())

if __name__ == "__main__":
    main()
