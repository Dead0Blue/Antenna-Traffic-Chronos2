# Antenna Traffic Forecasting with Chronos-2

This project uses **Chronos-2**, a state-of-the-art time series foundation model by Amazon Science, to predict cellular antenna traffic in a French city.

## Project Overview
The goal of this project is to evaluate the zero-shot forecasting capabilities of the **Chronos-2** model on real-world antenna traffic data. We use weekly traffic measurements from 86 antenna sectors to predict the final known measurement and evaluate the performance using **Root Mean Square Error (RMSE)**.

### Key Features
- **Data Preprocessing**: Robust parsing of French date formats and automatic regularization of irregular weekly frequencies (7D intervals).
- **Foundation Model**: Leverages `chronos-2-small`, a transformer-based model pretrained on billions of time series points.
- **Evaluation**: Zero-shot prediction comparison with actual ground truth.
- **Visualizations**: Performance dashboard showing correlation, error distribution, and individual sector trends.

## Results
- **RMSE**: **5.1189 Mbps**
- The model shows strong zero-shot performance, accurately capturing seasonal trends and traffic magnitudes without any sector-specific training.

### Performance Dashboard
![Optimization Dashboard](optimization_dashboard.png)

*The dashboard shows the high correlation between actual and predicted values (top-left) and sample forecasts for the highest-traffic sectors.*

## How to Run

### Prerequisites
- Python 3.9+
- Dependencies: `pandas`, `numpy`, `torch`, `chronos-forecasting`, `matplotlib`, `seaborn`, `scikit-learn`, `openpyxl`.

### Installation
```bash
pip install chronos-forecasting torch pandas matplotlib seaborn scikit-learn openpyxl
```

### Usage
1. **Forecast & Evaluate**: Run the main forecasting script to generate predictions.
   ```bash
   python forecast_traffic.py
   ```
   This will generate `prediction_results.csv`.

2. **Visualize Results**: Run the visualization script to generate the performance dashboard.
   ```bash
   python visualize_results.py
   ```

## Repository Structure
- `forecast_traffic.py`: Main pipeline for loading data, cleaning, and model inference.
- `visualize_results.py`: Script to generate the performance dashboard and plots.
- `explore_data.py`: Utility script for initial data exploration.
- `prediction_results.csv`: Final output comparing actual vs. predicted values.
- `actual_vs_predicted.png`: Correlation plot.
- `optimization_dashboard.png`: Full results dashboard.
- `histo_trafic.csv`: (Reference) Dataset of antenna traffic history.
- `DIM prédictif 2024 - Secteurs.xlsx`: (Reference) Metadata of antenna sectors.

## About Chronos-2
Chronos is a family of pretrained time series forecasting models that leverage language model architectures. A time series is transformed into a sequence of tokens via scaling and quantization, allowing the model to treat forecasting as a next-token prediction task.

For more information, visit the [Chronos GitHub repository](https://github.com/amazon-science/chronos-forecasting).
