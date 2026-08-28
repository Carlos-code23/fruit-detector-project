import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, UnidentifiedImageError

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
REPORT_PATH = Path("reports/cleaning_summary.json")

# 1. Definir las 10 frutas objetivo (Asegúrate de que coincidan con los nombres de carpetas en data/raw)
TARGET_FRUITS = {
    "Banana",
    "Apple Red 1",     
    "Apple Green 1",
    "Orange",
    "Lemon",
    "Strawberry",
    "Grape Red",
    "Pineapple",
    "Watermelon",
    "Peach"
}

def calculate_md5(file_path: Path) -> str:
    """Calcula el hash MD5 para detectar duplicados."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def process_single_image(img_info):
    """Procesa, valida y limpia una única imagen."""
    img_path, dest_path = img_info
    try:
        with Image.open(img_path) as img:
            img.verify()
        
        with Image.open(img_path) as img:
            img.load()
            file_hash = calculate_md5(img_path)
            
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest_path, "JPEG", quality=95)
            return {"status": "success", "hash": file_hash}
            
    except (UnidentifiedImageError, OSError, ValueError):
        return {"status": "corrupted", "hash": None}

def main():
    if not RAW_DATA_PATH.exists():
        print(f"Error: No se encontró la carpeta {RAW_DATA_PATH}. Descarga el dataset ahí primero.")
        return

    tasks = []
    print("🔎 Escaneando y filtrando carpetas objetivo...")

    # Buscar únicamente en las subcarpetas de las 10 frutas seleccionadas
    for split in ["Training", "Test"]:
        split_dir = RAW_DATA_PATH / split
        if not split_dir.exists():
            # Si el dataset no usa carpetas Training/Test a nivel superior, busca directamente
            split_dir = RAW_DATA_PATH

        for fruit_folder in split_dir.iterdir():
            if fruit_folder.is_dir() and fruit_folder.name in TARGET_FRUITS:
                for img_path in fruit_folder.glob("*.*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        rel_path = img_path.relative_to(RAW_DATA_PATH)
                        dest_path = PROCESSED_DATA_PATH / rel_path
                        tasks.append((img_path, dest_path))

    print(f"📦 Se encontraron {len(tasks)} imágenes pertenecientes a las 10 frutas.")
    print("⚡ Procesando imágenes en paralelo...")

    seen_hashes = set()
    stats = {
        "total_target_found": len(tasks),
        "copied_clean": 0,
        "duplicates_removed": 0,
        "corrupted_removed": 0
    }

    # Procesar usando múltiples hilos de la CPU para máxima velocidad
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single_image, tasks))

    for res in results:
        if res["status"] == "corrupted":
            stats["corrupted_removed"] += 1
        elif res["status"] == "success":
            h = res["hash"]
            if h in seen_hashes:
                stats["duplicates_removed"] += 1
            else:
                seen_hashes.add(h)
                stats["copied_clean"] += 1

    # Guardar reporte de auditoría
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(stats, f, indent=4)

    print("\n✅ Proceso finalizado. Resumen de la limpieza:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()