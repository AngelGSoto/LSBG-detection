'''
Script to dolownd images form Legacy
Based on the Amanda' script

'''
from pathlib import Path
from astropy.table import Table
import argparse
import os
import pandas as pd
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def read_table(file_name):
    try:
        if file_name.endswith('.ecsv'):
            data = Table.read(file_name, format="ascii.ecsv")
        else:
            data = pd.read_csv(file_name)
    except FileNotFoundError:
        logging.error("File not found.")
        return None
    return data

def download_legacy_image(url, file_path):
    if not os.path.exists(file_path):  # Check if the file already exists
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            logging.info(f"Downloaded: {file_path}")
        else:
            logging.error(f"Failed to download: {url}")
    else:
        logging.info(f"Skipping {file_path}, already downloaded.")

def download_legacy(data, out_path, radii_default="default_value_for_radii", checkpoint_file='download_checkpoint.txt'):
    downloaded_files = {}

    # Check if checkpoint file exists and load downloaded filenames
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as file:
            for line in file:
                downloaded_files[line.strip()] = True

    for idx, row in data.iterrows():
        ra = row["ra"]  # Update column name if needed
        dec = row["dec"]  # Update column name if needed
        name = f"{ra}_{dec}"  # Creating a name using RA and DEC

        if 'radii' in row:
            radii = row['radii']
        else:
            radii = radii_default

        file_name = f'{out_path}/{name}_{radii}pix.jpeg'
        logging.info(f"Downloading image for {name}")
        url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&size={radii}&layer=ls-dr9&pixscale=0.262&bands=grz"

        if file_name not in downloaded_files:
            download_legacy_image(url, file_name)
            downloaded_files[file_name] = True

            # Update checkpoint file
            with open(checkpoint_file, 'a') as checkpoint:
                checkpoint.write(file_name + '\n')

def main():
    parser = argparse.ArgumentParser(description="Download images from Legacy")
    parser.add_argument("table", type=str, help="Name of table, taken the prefix ")
    parser.add_argument("--Object", type=str, default=None, help="Id object of a given source")
    parser.add_argument("--legacy", action="store_true", help="make legacy images")
    parser.add_argument("--radii_default", type=str, default="default_value_for_radii", help="Default value for radii if 'radii' column is not found")

    args = parser.parse_args()
    file_name = args.table + (".ecsv" if os.path.exists(args.table + ".ecsv") else ".csv")
    data = read_table(file_name)

    if args.Object is not None:
        object_id = str(args.Object)
        mask = np.array([source in object_id for source in data["RA"]])
        data = data[mask]

    if args.legacy:
        dir_output = Path(".")
        path_legacy = dir_output / 'legacy_color_images'

        if not path_legacy.exists():
            path_legacy.mkdir(parents=True, exist_ok=True)
            logging.info(f"Directory '{path_legacy}' created!")
        else:
            logging.info(f"Directory '{path_legacy}' already exists!")

        download_legacy(data, path_legacy, args.radii_default)
        logging.info("\nDownload from Legacy Survey finished!")

if __name__ == "__main__":
    main()
