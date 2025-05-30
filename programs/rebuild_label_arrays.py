import numpy as np
import os
import pandas as pd
import re
from tqdm import tqdm

# Configuración
JPEG_DIRS = {
    'train': '../Datasets_DeepShadows/Jpeg_data/Training/',
    'val': '../Datasets_DeepShadows/Jpeg_data/Validation/',
    'test': '../Datasets_DeepShadows/Jpeg_data/Test/'
}

OUTPUT_DIR = '../Datasets_DeepShadows/Galaxies_data/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar catálogos de referencia
LSB_PATH = '../Datasets_DeepShadows/Datasets/random_LSBGs_all.csv'
ARTIFACT_PATH = '../Datasets_DeepShadows/Datasets/random_negative_all_2.csv'

lsb_df = pd.read_csv(LSB_PATH)
art_df = pd.read_csv(ARTIFACT_PATH)

# Función para parsear nombres de archivo
def parse_filename(filename):
    match = re.match(r'^([\d\.\-]+)_([\d\.\-]+)_\d+_256pix\.jpe?g', filename, re.IGNORECASE)
    if match:
        try:
            ra = float(match.group(1))
            dec = float(match.group(2))
            return (ra, dec)
        except ValueError:
            return (None, None)
    return (None, None)

# Función para obtener etiqueta
def get_label(ra, dec):
    if ra is None or dec is None:
        return 0
    
    # Buscar en galaxias LSB
    match_lsb = lsb_df[
        (abs(lsb_df['ra'] - ra) < 0.001) & 
        (abs(lsb_df['dec'] - dec) < 0.001)
    ]
    if not match_lsb.empty:
        return 1
    
    # Buscar en artefactos
    match_art = art_df[
        (abs(art_df['ra'] - ra) < 0.001) & 
        (abs(art_df['dec'] - dec) < 0.001)
    ]
    if not match_art.empty:
        return 0
    
    return 0

for set_name, jpeg_dir in JPEG_DIRS.items():
    print(f"\nProcesando conjunto: {set_name}")
    
    # Obtener archivos ordenados
    files = sorted([f for f in os.listdir(jpeg_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Asignar etiquetas
    labels = []
    for filename in tqdm(files, desc="Asignando etiquetas"):
        ra, dec = parse_filename(filename)
        label = get_label(ra, dec)
        labels.append(label)
    
    # Guardar array
    labels_array = np.array(labels, dtype=np.int32)
    output_path = os.path.join(OUTPUT_DIR, f'y_{set_name}.npy')
    np.save(output_path, labels_array)
    
    # Estadísticas
    galaxy_count = np.sum(labels_array == 1)
    print(f"Guardado {output_path} con {len(labels_array)} etiquetas")
    print(f"Galaxias: {galaxy_count} ({galaxy_count/len(labels_array):.2%})")
    print(f"Artefactos: {len(labels_array) - galaxy_count}")
