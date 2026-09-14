import pandas as pd
import zipfile

def load_data() -> pd.DataFrame:
    # Pandas will automatically decompress and read the CSV inside
    zip_path = "data/raw/airline_loyalty.zip"
    
    # Open the zip archive
    with zipfile.ZipFile(zip_path, 'r') as archive:
        # Target the specific CSV file inside the archive
        with archive.open('Customer Loyalty History.csv') as file:
            df = pd.read_csv(file)

            
    return df