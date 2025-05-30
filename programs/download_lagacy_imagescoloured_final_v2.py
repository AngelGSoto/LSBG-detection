from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import pandas as pd
import requests
import logging
import argparse
import time
import psutil
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configuración avanzada de gestión de recursos
class ResourceManager:
    def __init__(self, priority=0.5):
        """
        priority: 0.1 (baja) - 1.0 (alta)
        """
        self.priority = max(0.1, min(1.0, priority))
        self.last_check = time.time()
        self.cooldown = 0
        
    def can_proceed(self):
        """Determina si es seguro continuar con las descargas"""
        # Enfriamiento después de pausas forzadas
        if self.cooldown > time.time():
            return False
        
        # Solo verificar cada 15 segundos
        if time.time() - self.last_check < 15:
            return True
            
        self.last_check = time.time()
        
        # Obtener métricas del sistema
        cpu_percent = psutil.cpu_percent(interval=1)
        mem_available = psutil.virtual_memory().available / (1024 ** 3)  # GB
        
        # Umbrales ajustables por prioridad
        cpu_threshold = 80 - (20 * self.priority)  # Más estricto con prioridad baja
        mem_threshold = 1.0 + (2 * self.priority)  # Más memoria libre con prioridad baja
        
        # Verificar condiciones
        if cpu_percent > cpu_threshold:
            logging.warning(f"CPU usage high ({cpu_percent}% > {cpu_threshold}%), pausing downloads")
            self.cooldown = time.time() + 30  # Pausa de 30 segundos
            return False
            
        if mem_available < mem_threshold:
            logging.warning(f"Memory low ({mem_available:.1f}GB < {mem_threshold:.1f}GB), pausing downloads")
            self.cooldown = time.time() + 45  # Pausa de 45 segundos
            return False
            
        return True
    
    def adaptive_sleep(self):
        """Pausa adaptativa basada en condiciones del sistema"""
        if not self.can_proceed():
            sleep_time = min(60, 15 + (1 - self.priority) * 45)
            logging.info(f"System busy. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            return True
        return False

def read_table(file_name):
    try:
        if file_name.endswith('.ecsv'):
            from astropy.table import Table
            data = Table.read(file_name, format="ascii.ecsv")
            return data.to_pandas()
        else:
            return pd.read_csv(file_name)
    except FileNotFoundError:
        logging.error("File not found.")
        return None

def download_legacy_image(url_file_path, resource_manager):
    """Función de descarga con gestión de recursos"""
    url, file_path = url_file_path
    
    # Esperar si el sistema está ocupado
    if resource_manager.adaptive_sleep():
        return None  # Saltar esta descarga por ahora
    
    if not os.path.exists(file_path):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                logging.info(f"Downloaded: {file_path}")
                return file_path
            else:
                logging.error(f"Failed to download: {url} (Status: {response.status_code})")
        except requests.exceptions.RequestException as e:
            logging.error(f"Download error for {url}: {str(e)}")
    else:
        logging.info(f"Skipping {file_path}, already exists.")
    
    return None

def download_legacy(data, out_path, radii_default=256, checkpoint_file=None, priority=0.5):
    # Crear directorio si no existe
    os.makedirs(out_path, exist_ok=True)
    
    # Usar checkpoint en el directorio de salida
    if checkpoint_file is None:
        checkpoint_file = os.path.join(out_path, 'download_checkpoint.txt')
    
    # Inicializar gestor de recursos
    resource_manager = ResourceManager(priority=priority)
    
    # Cargar archivos ya descargados
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

    # Preparar lista de descargas
    for idx, row in data.iterrows():
        ra = row["ra"]
        dec = row["dec"]
        radii = row.get('radii', radii_default)
        
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
    
    # Configuración dinámica de workers
    def calculate_workers():
        """Calcula workers óptimos basado en uso del sistema"""
        if resource_manager.priority < 0.3:
            return 2  # Mínimo para prioridad baja
        elif resource_manager.priority < 0.7:
            return 4  # Balanceado
        else:
            return 6  # Máximo para prioridad alta

    # Ejecutar descargas con gestión de recursos
    downloaded_count = 0
    with ThreadPoolExecutor(max_workers=calculate_workers()) as executor:
        futures = {executor.submit(download_legacy_image, item, resource_manager): item 
                   for item in urls_file_paths}
        
        # Procesar resultados con barra de progreso
        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded_count += 1
                # Actualizar checkpoint inmediatamente
                with open(checkpoint_file, 'a') as f:
                    f.write(result + '\n')
                
                # Actualizar contador cada 10 descargas
                if downloaded_count % 10 == 0:
                    logging.info(f"Progress: {downloaded_count}/{len(urls_file_paths)} downloaded")
                    
                # Verificar recursos con más frecuencia bajo carga
                if downloaded_count % 50 == 0:
                    resource_manager.adaptive_sleep()

    logging.info(f"Successfully downloaded {downloaded_count} images")

def main():
    parser = argparse.ArgumentParser(description="Download images from Legacy")
    parser.add_argument("table", help="Path to input table")
    parser.add_argument("--object", help="Specific object ID to download")
    parser.add_argument("--legacy", action="store_true", help="Download legacy images")
    parser.add_argument("--radii_default", type=int, default=256, help="Default pixel radius")
    parser.add_argument("--output", default="./legacy_color_images", help="Output directory")
    parser.add_argument("--priority", type=float, default=0.5, 
                        help="Task priority (0.1=low, 1.0=high, default=0.5)")
    
    args = parser.parse_args()
    
    data = read_table(args.table)
    if data is None:
        return

    if args.object:
        # Asumiendo que tienes una columna 'object_id' - ajusta según sea necesario
        data = data[data['object_id'] == args.object]

    if args.legacy:
        download_legacy(data, args.output, args.radii_default, priority=args.priority)

if __name__ == "__main__":
    # Verificar si psutil está instalado
    try:
        import psutil
    except ImportError:
        print("Error: psutil library is required. Install with: pip install psutil")
        sys.exit(1)
    
    main()
