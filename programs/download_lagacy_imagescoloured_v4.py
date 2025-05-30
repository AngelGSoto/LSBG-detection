import time
from pathlib import Path
from astropy.table import Table
import argparse
import os
import pandas as pd
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
failed_objects = []

def read_table(file_name):
    try:
        if file_name.endswith('.ecsv'):
            data = Table.read(file_name, format="ascii.ecsv").to_pandas()
        else:
            data = pd.read_csv(file_name)
    except FileNotFoundError:
        logging.error("File not found.")
        return None
    return data

def download_legacy_image(url, file_path):
    try:
        if not os.path.exists(file_path):
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                logging.info(f"Downloaded: {file_path}")
                return True
            else:
                logging.error(f"Failed to download: {url}")
                logging.error(f"Failed response code: {response.status_code}")
                return False
        else:
            logging.info(f"Skipping {file_path}, already downloaded.")
            return True
    except Exception as e:
        logging.error(f"Error during download: {e}")
        return False

def save_failed_objects(failed_objects_list, data, output_file='failed_objects.csv'):
    try:
        failed_data = data[data['Name'].isin(failed_objects_list)][['ra', 'dec']]
        failed_data.to_csv(output_file, index=False)
        logging.info(f"Failed objects list saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving failed objects: {e}")

def download_legacy(data, out_path, radii_default="default_value_for_radii", checkpoint_file='download_checkpoint.txt'):
    downloaded_files = {}
    message_limit = 1000  # Adjust this limit as needed
    msg_count = 0
    start_time = time.time()

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as file:
            for line in file:
                downloaded_files[line.strip()] = True

    for idx, row in data.iterrows():
        ra = row["ra"]
        dec = row["dec"]
        name = f"{ra}_{dec}"

        if 'radii' in row:
            radii = row['radii']
        else:
            radii = radii_default

        file_name = f'{out_path}/{name}_{radii}pix.jpeg'
        logging.info(f"Downloading image for {name}")
        url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&size={radii}&layer=ls-dr9&pixscale=0.262&bands=grz"

        try:
            if file_name not in downloaded_files:
                success = download_legacy_image(url, file_name)
                downloaded_files[file_name] = success

                if not success:
                    failed_objects.append(row['Name'])  # Record failed objects

                with open(checkpoint_file, 'a') as checkpoint:
                    checkpoint.write(file_name + '\n')

                # Check message rate limit
                msg_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1:  # Adjust this time window as needed
                    msg_count = 0
                    start_time = time.time()
                if msg_count >= message_limit:
                    time.sleep(1)  # Sleep for 1 second if approaching the message limit

        except Exception as e:
            logging.error(f"Error processing object {name}: {e}")
            failed_objects.append(row['Name'])  # Record failed objects

    logging.info("\nDownload from Legacy Survey finished!")
    if failed_objects:
        save_failed_objects(failed_objects, data)

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
        mask = np.array([source in object_id for source in data["ra"]])
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
