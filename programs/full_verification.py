import numpy as np
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import warnings

# Ignorar warnings de imágenes
warnings.filterwarnings('ignore', category=UserWarning)

# Configuración de rutas
BASE_DIR = '../Datasets_DeepShadows/'
ARRAY_DIR = os.path.join(BASE_DIR, 'array_images/')
LABEL_DIR = os.path.join(BASE_DIR, 'Galaxies_data/')
JPEG_DIRS = {
    'train': os.path.join(BASE_DIR, 'Jpeg_data/Training/'),
    'val': os.path.join(BASE_DIR, 'Jpeg_data/Validation/'),
    'test': os.path.join(BASE_DIR, 'Jpeg_data/Test/')
}

# Catálogos de referencia
LSB_PATH = os.path.join(BASE_DIR, 'Datasets/random_LSBGs_all.csv')
ARTIFACT_PATH = os.path.join(BASE_DIR, 'Datasets/random_negative_all_2.csv')

# Cargar catálogos
print("Cargando catálogos de referencia...")
lsb_df = pd.read_csv(LSB_PATH)
art_df = pd.read_csv(ARTIFACT_PATH)
print(f"Catálogo LSB: {len(lsb_df)} objetos")
print(f"Catálogo Artefactos: {len(art_df)} objetos")

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

# Función para buscar en catálogos
def find_in_catalogs(ra, dec, tol=0.001):
    """Busca coordenadas en los catálogos con tolerancia"""
    results = {
        'in_lsb': False,
        'in_artifacts': False,
        'expected_label': 0
    }
    
    if ra is None or dec is None:
        return results
    
    # Buscar en LSB
    lsb_match = lsb_df[
        (np.abs(lsb_df['ra'] - ra) < tol) & 
        (np.abs(lsb_df['dec'] - dec) < tol)
    ]
    if not lsb_match.empty:
        results['in_lsb'] = True
        results['expected_label'] = 1
    
    # Buscar en artefactos
    art_match = art_df[
        (np.abs(art_df['ra'] - ra) < tol) & 
        (np.abs(art_df['dec'] - dec) < tol)
    ]
    if not art_match.empty:
        results['in_artifacts'] = True
        # Solo sobrescribir si no está en LSB (prioridad galaxias)
        if not results['in_lsb']:
            results['expected_label'] = 0
    
    return results

def verify_dataset(set_name, num_samples=10):
    print(f"\n{'='*50}")
    print(f"Verificando conjunto: {set_name}")
    print(f"{'='*50}")
    
    # 1. Cargar arrays
    try:
        X = np.load(os.path.join(ARRAY_DIR, f'X_{set_name}.npy'))
        y = np.load(os.path.join(LABEL_DIR, f'y_{set_name}.npy'))
        print(f"Arrays cargados: X.shape={X.shape}, y.shape={y.shape}")
    except Exception as e:
        print(f"Error cargando arrays: {str(e)}")
        return
    
    # 2. Obtener lista de archivos JPEG
    jpeg_dir = JPEG_DIRS[set_name]
    try:
        jpeg_files = sorted([f for f in os.listdir(jpeg_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Archivos JPEG encontrados: {len(jpeg_files)}")
    except Exception as e:
        print(f"Error leyendo directorio JPEG: {str(e)}")
        return
    
    # 3. Verificar correspondencia de tamaños
    if len(X) != len(y) or len(X) != len(jpeg_files):
        print(f"¡ERROR! Tamaños no coinciden:")
        print(f"  Array imágenes: {len(X)}")
        print(f"  Array etiquetas: {len(y)}")
        print(f"  Archivos JPEG: {len(jpeg_files)}")
        return
    
    print("✓ Tamaños coinciden")
    
    # 4. Verificación detallada de muestras aleatorias
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), num_samples, replace=False)
    
    print(f"\nVerificando {num_samples} muestras aleatorias:")
    for i, idx in enumerate(sample_indices):
        filename = jpeg_files[idx]
        ra, dec = parse_filename(filename)
        catalog_info = find_in_catalogs(ra, dec)
        
        # Resultados
        status_label = "✓" if y[idx] == catalog_info['expected_label'] else "✗"
        status_lsb = "✓" if catalog_info['in_lsb'] else "✗"
        status_art = "✓" if catalog_info['in_artifacts'] else "✗"
        
        print(f"\nMuestra {i+1}: Índice {idx} - {filename}")
        print(f"  Coordenadas: RA={ra:.6f}, DEC={dec:.6f}")
        print(f"  En catálogo LSB: {status_lsb} | En catálogo Artefactos: {status_art}")
        print(f"  Etiqueta esperada: {catalog_info['expected_label']} | Etiqueta real: {y[idx]} {status_label}")
        
        # Visualización
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # Imagen del array
        axs[0].imshow(X[idx])
        axs[0].set_title(f"Array X[{idx}]\nLabel: {y[idx]}")
        axs[0].axis('off')
        
        # Información de catálogos
        catalog_text = (
            f"En LSB: {catalog_info['in_lsb']}\n"
            f"En Artefactos: {catalog_info['in_artifacts']}\n"
            f"Label esperado: {catalog_info['expected_label']}"
        )
        
        axs[1].text(0.5, 0.5, catalog_text, 
                   ha='center', va='center', fontsize=12)
        axs[1].axis('off')
        axs[1].set_title("Información de Catálogos")
        
        plt.tight_layout()
        plt.show()
    
    # 5. Verificación completa de etiquetas
    print("\nVerificando todas las etiquetas...")
    correct_labels = 0
    in_lsb_count = 0
    in_art_count = 0
    in_both_count = 0
    in_neither_count = 0
    
    for idx in tqdm(range(len(X)), desc="Progreso"):
        filename = jpeg_files[idx]
        ra, dec = parse_filename(filename)
        catalog_info = find_in_catalogs(ra, dec)
        
        # Contar ocurrencias en catálogos
        if catalog_info['in_lsb'] and catalog_info['in_artifacts']:
            in_both_count += 1
        elif catalog_info['in_lsb']:
            in_lsb_count += 1
        elif catalog_info['in_artifacts']:
            in_art_count += 1
        else:
            in_neither_count += 1
        
        # Verificar etiqueta
        if y[idx] == catalog_info['expected_label']:
            correct_labels += 1
    
    # 6. Reporte final
    print("\n" + "="*50)
    print("Reporte de Verificación Final")
    print("="*50)
    print(f"Total muestras: {len(X)}")
    print(f"Etiquetas correctas: {correct_labels} ({correct_labels/len(X):.2%})")
    print("\nDistribución en catálogos:")
    print(f"- Solo en LSB: {in_lsb_count} ({in_lsb_count/len(X):.2%})")
    print(f"- Solo en Artefactos: {in_art_count} ({in_art_count/len(X):.2%})")
    print(f"- En ambos catálogos: {in_both_count} ({in_both_count/len(X):.2%})")
    print(f"- En ningún catálogo: {in_neither_count} ({in_neither_count/len(X):.2%})")
    
    # 7. Verificar distribución de etiquetas
    galaxy_count = np.sum(y == 1)
    artifact_count = np.sum(y == 0)
    print("\nDistribución de etiquetas en el array:")
    print(f"- Galaxias (1): {galaxy_count} ({galaxy_count/len(X):.2%})")
    print(f"- Artefactos (0): {artifact_count} ({artifact_count/len(X):.2%})")
    
    # 8. Guardar resultados detallados
    results = []
    for idx in range(len(X)):
        filename = jpeg_files[idx]
        ra, dec = parse_filename(filename)
        catalog_info = find_in_catalogs(ra, dec)
        correct = y[idx] == catalog_info['expected_label']
        
        results.append({
            'filename': filename,
            'ra': ra,
            'dec': dec,
            'label': y[idx],
            'expected_label': catalog_info['expected_label'],
            'in_lsb': catalog_info['in_lsb'],
            'in_artifacts': catalog_info['in_artifacts'],
            'correct': correct
        })
    
    results_df = pd.DataFrame(results)
    results_path = os.path.join(LABEL_DIR, f'verification_results_{set_name}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResultados detallados guardados en: {results_path}")

# Ejecutar verificación para todos los conjuntos
print("\n" + "="*50)
print("INICIANDO VERIFICACIÓN COMPLETA DE DATASETS")
print("="*50 + "\n")

for dataset in ['train', 'val', 'test']:
    verify_dataset(dataset)
    print("\n" + "="*100 + "\n")

print("Verificación completada para todos los conjuntos!")
