import os
import pandas as pd
import zipfile

def load_data() -> pd.DataFrame:
    """Loads dataset dynamically, supporting multi-file ZIP archives."""
    
    file_path = os.getenv("INPUT_DATA_PATH")
    
    if not file_path:
        raise ValueError("INPUT_DATA_PATH is not set in the .env file.")
        
    # Resolve absolute path and normalize slashes
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_path = os.path.normpath(os.path.join(project_root, file_path))
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data file not found at: {full_path}")
        
    print(f"Loading data from: {full_path}")
    
    _, file_extension = os.path.splitext(full_path)
    file_extension = file_extension.lower()
    
    if file_extension == '.csv':
        return pd.read_csv(full_path)
        
    elif file_extension == '.zip':
        internal_file = os.getenv("ZIP_INTERNAL_FILE_NAME")
        
        # Open the zip archive to inspect its contents
        with zipfile.ZipFile(full_path, 'r') as archive:
            available_files = archive.namelist()
            
            # Scenario A: Only one file in the zip (Pandas can handle safely)
            if len(available_files) == 1:
                with archive.open(available_files[0]) as target_csv:
                    return pd.read_csv(target_csv)
            
            # Scenario B: Multiple files (requires .env specification)
            else:
                if not internal_file:
                    raise ValueError(f"Multiple files detected in ZIP: {available_files}. "
                                     "Please specify 'ZIP_INTERNAL_FILE_NAME' in your .env file.")
                                     
                if internal_file not in available_files:
                    raise ValueError(f"The file '{internal_file}' was not found inside the ZIP archive. "
                                     f"Available files: {available_files}")
                
                # Extract and read the specific file requested
                print(f"Extracting '{internal_file}' from archive...")
                with archive.open(internal_file) as target_csv:
                    return pd.read_csv(target_csv)
    else:
        raise ValueError(f"Unsupported format: '{file_extension}'. Only .csv and .zip are supported.")