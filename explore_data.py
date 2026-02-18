import pandas as pd
import locale

def parse_french_date(date_str):
    # Example: "lundi 18 juin 2018"
    # Remove day of week (lundi, mardi, etc.)
    parts = date_str.split()
    if len(parts) >= 4:
        # 18 juin 2018
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
    return date_str

def load_data(file_path):
    # Read the file to find where the header is
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
    
    # Filter out empty rows or rows that might repeat the header
    df = df[df['secteur'] != 'secteur']
    
    # Parse dates
    df['date'] = df['tstamp'].apply(lambda x: parse_french_date(str(x)))
    # Remove any rows where date couldn't be parsed (NaT)
    df = df.dropna(subset=['tstamp'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Convert traffic to numeric
    df['trafic_mbps'] = pd.to_numeric(df['trafic_mbps'], errors='coerce')
    df = df.dropna(subset=['trafic_mbps'])
    
    return df

if __name__ == "__main__":
    path = r'c:\Users\PC\Desktop\Supélec\S8\Projet S8\document 1\histo_trafic.csv'
    df = load_data(path)
    print("Data Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())
    print("\nValue counts for sectors (top 5):")
    print(df['secteur'].value_counts().head())
    
    # Check for number of unique measurements per sector
    sector_counts = df['secteur'].value_counts()
    print(f"\nMean measurements per sector: {sector_counts.mean()}")
    print(f"Min measurements per sector: {sector_counts.min()}")
    print(f"Max measurements per sector: {sector_counts.max()}")
    
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    
    # Check one sector to see the frequency
    one_sector = df[df['secteur'] == df['secteur'].iloc[0]].sort_values('date')
    print(f"\nSample sector {df['secteur'].iloc[0]} frequency check:")
    print(one_sector['date'].diff().value_counts())
