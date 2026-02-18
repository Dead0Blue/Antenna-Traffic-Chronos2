import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error

def parse_french_date(date_str):
    parts = str(date_str).split()
    if len(parts) >= 4:
        day = parts[1]
        month_fr = parts[2].lower()
        year = parts[3]
        months = {'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04', 'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08', 'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'}
        return f"{year}-{months.get(month_fr, '01')}-{day.zfill(2)}"
    return None

def load_raw_data(file_path):
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
    df['date'] = pd.to_datetime(df['tstamp'].apply(parse_french_date), errors='coerce')
    df['trafic_mbps'] = pd.to_numeric(df['trafic_mbps'], errors='coerce')
    df = df.dropna(subset=['date', 'trafic_mbps', 'secteur'])
    return df

def main():
    # Load results
    results_df = pd.read_csv('prediction_results.csv')
    rmse = np.sqrt(mean_squared_error(results_df['actual'], results_df['predicted']))
    results_df['error'] = results_df['actual'] - results_df['predicted']
    
    # Setup the figure
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Scatter Plot
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=results_df, x='actual', y='predicted', alpha=0.7, ax=ax1, color='royalblue')
    max_val = max(results_df['actual'].max(), results_df['predicted'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_title(f'Actual vs Predicted Traffic (RMSE: {rmse:.2f})', fontsize=14)
    ax1.set_xlabel('Actual Traffic (Mbps)')
    ax1.set_ylabel('Predicted Traffic (Mbps)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Error Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(results_df['error'], kde=True, ax=ax2, color='coral')
    ax2.set_title('Error Distribution (Actual - Predicted)', fontsize=14)
    ax2.set_xlabel('Prediction Error (Mbps)')
    
    # 3. Time Series Samples
    # Load original data for trend visualization
    print("Loading original history for detailed plots...")
    raw_df = load_raw_data('histo_trafic.csv')
    
    # Select 4 sectors to showcase (mix of good and average)
    sample_sectors = results_df.sort_values('actual', ascending=False)['secteur'].head(4).tolist()
    
    for i, sector in enumerate(sample_sectors):
        ax = fig.add_subplot(gs[1 + (i // 2), i % 2])
        
        sector_history = raw_df[raw_df['secteur'] == sector].sort_values('date')
        # Filter to last 20 weeks for visibility
        plot_history = sector_history.tail(20)
        
        pred_val = results_df[results_df['secteur'] == sector]['predicted'].values[0]
        actual_val = results_df[results_df['secteur'] == sector]['actual'].values[0]
        
        # Plot history (excluding the very last point which was predicted)
        ax.plot(plot_history.iloc[:-1]['date'], plot_history.iloc[:-1]['trafic_mbps'], 
                marker='o', linestyle='-', color='gray', alpha=0.5, label='Context')
        
        # Plot last point (actual)
        ax.scatter(plot_history.iloc[-1]['date'], actual_val, color='green', s=100, label='Actual Last', zorder=5)
        # Plot last point (predicted)
        ax.scatter(plot_history.iloc[-1]['date'], pred_val, color='red', marker='x', s=100, label='Predicted Last', zorder=5)
        
        ax.set_title(f'Sector: {sector}', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.savefig('optimization_dashboard.png')
    print("Dashboard saved as optimization_dashboard.png")

if __name__ == "__main__":
    main()
