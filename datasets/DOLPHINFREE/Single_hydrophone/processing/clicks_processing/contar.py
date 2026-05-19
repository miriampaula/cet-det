# count_clicks.py
import argparse
import sys
from pathlib import Path
import numpy as np
import csv

def contar_clicks_en_npy(path: Path) -> int:
    """
    Devuelve el número de clicks en un .npy.
    - Si el array parece un mapa binario (0/1), cuenta la suma.
    - En caso contrario, cuenta el nº de elementos (típico para arrays de índices).
    """
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"No se pudo leer {path.name}: {e}")

    # Aplanar por si llega con dimensiones raras/objetos
    try:
        a = np.array(arr).ravel()
    except Exception:
        # Último recurso: contar elementos iterables
        try:
            return sum(1 for _ in arr)
        except Exception as e:
            raise RuntimeError(f"Formato no soportado en {path.name}: {e}")

    if a.size == 0:
        return 0

    # Heurística: ¿es un mapa 0/1?
    amin, amax = np.min(a), np.max(a)
    es_binario = np.all((a == 0) | (a == 1)) or (amin >= 0 and amax <= 1 and a.dtype == np.bool_)
    if es_binario:
        return int(a.sum())

    # Por defecto: array de índices → nº de filas
    return int(a.size)

def main():
    p = argparse.ArgumentParser(description="Cuenta clicks en ficheros .npy")
    p.add_argument("carpeta", type=Path, help="Carpeta con .npy")
    p.add_argument("--patron", default="*.npy",
                   help="Patrón glob (por ej. '*_peaks*.npy'). Por defecto: *.npy")
    p.add_argument("--recursivo", action="store_true", help="Buscar recursivamente")
    p.add_argument("--csv", type=Path, default=None, help="Ruta para exportar CSV resumen")
    args = p.parse_args()

    if not args.carpeta.is_dir():
        print(f"ERROR: {args.carpeta} no es una carpeta", file=sys.stderr)
        sys.exit(1)

    files = list(args.carpeta.rglob(args.patron) if args.recursivo else args.carpeta.glob(args.patron))
    files = [f for f in files if f.is_file()]

    if not files:
        print("No se encontraron .npy con ese patrón.")
        sys.exit(0)

    total = 0
    filas_csv = []

    print(f"Analizando {len(files)} archivo(s)...\n")
    for f in sorted(files):
        try:
            n = contar_clicks_en_npy(f)
        except Exception as e:
            print(f"[FALLO] {f}: {e}")
            continue
        total += n
        filas_csv.append((str(f), n))
        print(f"{f.name:50s}  ->  {n:6d} clicks")

    print("\n" + "-"*64)
    print(f"TOTAL CLICKS: {total}")

    if args.csv:
        try:
            with open(args.csv, "w", newline="", encoding="utf-8") as out:
                w = csv.writer(out)
                w.writerow(["fichero", "clicks"])
                w.writerows(filas_csv)
                w.writerow(["TOTAL", total])
            print(f"Resumen guardado en: {args.csv}")
        except Exception as e:
            print(f"No se pudo escribir el CSV: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
