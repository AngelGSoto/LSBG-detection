import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# Configuración
INPUT_DIRS = {
    'train': '../Datasets_DeepShadows/Jpeg_data/Training/',
    'val': '../Datasets_DeepShadows/Jpeg_data/Validation/',
    'test': '../Datasets_DeepShadows/Jpeg_data/Test/'
}

OUTPUT_DIR = '../Datasets_DeepShadows/array_images/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros de procesamiento
TARGET_SIZE = (64, 64)
CROP_FACTOR = 0.2  # Recortar 20% de cada borde

def process_image(img_path):
    img = Image.open(img_path)
    
    # Recorte proporcional
    width, height = img.size
    crop_w = int(width * CROP_FACTOR)
    crop_h = int(height * CROP_FACTOR)
    
    img = img.crop((
        crop_w, 
        crop_h, 
        width - crop_w, 
        height - crop_h
    ))
    
    # Redimensionar
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    
    # Convertir a array y normalizar
    return np.array(img) / 255.0

for set_name, input_dir in INPUT_DIRS.items():
    print(f"\nProcesando conjunto: {set_name}")
    
    # Obtener archivos ordenados
    files = sorted([f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Procesar imágenes
    images = []
    for filename in tqdm(files, desc="Procesando imágenes"):
        img_path = os.path.join(input_dir, filename)
        img_array = process_image(img_path)
        images.append(img_array)
    
    # Guardar array
    images_array = np.array(images, dtype=np.float32)
    output_path = os.path.join(OUTPUT_DIR, f'X_{set_name}.npy')
    np.save(output_path, images_array)
    print(f"Guardado {output_path} con {len(images_array)} imágenes")
