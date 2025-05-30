from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import pandas as pd
import requests
import logging
import numpy as np  # Añadido para manejar máscaras

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def read_table(file_name):
    try:
        if file_name.endswith('.ecsv'):
            from astropy.table import Table
            data = Table.read(file_name, format="ascii.ecsv")
            return data.to_pandas()  # Convertir a DataFrame
        else:
            return pd.read_csv(file_name)
    except FileNotFoundError:
        logging.error("File not found.")
        return None

def download_legacy_image(url_file_path):
    url, file_path = url_file_path
    if not os.path.exists(file_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            logging.info(f"Downloaded: {file_path}")
            return file_path
        else:
            logging.error(f"Failed to download: {url} (Status: {response.status_code})")
            return None
    else:
        logging.info(f"Skipping {file_path}, already exists.")
        return file_path

def download_legacy(data, out_path, radii_default=256, checkpoint_file=None):
    # Crear directorio si no existe
    os.makedirs(out_path, exist_ok=True)
    
    # Usar checkpoint en el directorio de salida
    if checkpoint_file is None:
        checkpoint_file = os.path.join(out_path, 'download_checkpoint.txt')
    
    downloaded_files = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            downloaded_files = {line.strip(): True for line in f}
        logging.info(f"Checkpoint loaded with {len(downloaded_files)} entries")

    urls_file_paths = []
    missing = []
    
    # Verificar columnas requeridas
    if 'ra' not in data.columns or 'dec' not in data.columns:
        logging.error("Dataframe missing 'ra' or 'dec' columns")
        return

    for idx, row in data.iterrows():
        ra = row["ra"]
        dec = row["dec"]
        radii = row['radii'] if 'radii' in row else radii_default
        
        # Generar nombre único con índice
        unique_name = f"{ra}_{dec}_{idx}"
        file_name = f"{unique_name}_{radii}pix.jpeg"
        file_path = os.path.join(out_path, file_name)
        
        url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&size={radii}&layer=ls-dr9&pixscale=0.262&bands=grz"
        
        if file_path not in downloaded_files and not os.path.exists(file_path):
            urls_file_paths.append((url, file_path))
        else:
            missing.append(file_path)

    logging.info(f"Total images to download: {len(urls_file_paths)}")
    logging.info(f"Images already downloaded: {len(missing)}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(download_legacy_image, urls_file_paths))

    # Actualizar checkpoint
    with open(checkpoint_file, 'a') as f:
        for result in results:
            if result:
                f.write(result + '\n')

    logging.info(f"Successfully downloaded {len([r for r in results if r])} images")

def main():
    parser = argparse.ArgumentParser(description="Download images from Legacy")
    parser.add_argument("table", help="Path to input table")
    parser.add_argument("--object", help="Specific object ID to download")
    parser.add_argument("--legacy", action="store_true", help="Download legacy images")
    parser.add_argument("--radii_default", type=int, default=128, help="Default pixel radius")
    parser.add_argument("--output", default="./legacy_color_images", help="Output directory")
    
    args = parser.parse_args()
    
    data = read_table(args.table)
    if data is None:
        return

    if args.object:
        data = data[data['object_id'] == args.object]  # Ajustar columna ID según tu dataset

    if args.legacy:
        download_legacy(data, args.output, args.radii_default)

if __name__ == "__main__":
    import argparse
    main()
