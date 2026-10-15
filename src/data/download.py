"""O.E.2 - Descarga del dataset SVHN (Sprint 1).

Descargador robusto (con reintentos y reanudacion via cabecera HTTP ``Range``)
para los archivos ``.mat`` oficiales de Stanford. Se descargan los splits
``train`` y ``test`` (el split ``extra`` ~530k imagenes se omite para la demo).
Verifica el MD5 de cada archivo, por lo que ``torchvision.datasets.SVHN``
posteriormente los reconoce como validos.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

from ..config import load_config, resolve
from ..utils import ensure_dir

BASE_URL = "http://ufldl.stanford.edu/housenumbers/"
SPLITS = {
    "train": ("train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"),
    "test": ("test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"),
}
CHUNK = 1 << 16
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; svhn-project/1.0)"}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download_one(url: str, dest: Path, md5sum: str, retries: int = 10) -> bool:
    for attempt in range(1, retries + 1):
        try:
            if dest.exists() and _md5(dest) == md5sum:
                return True

            pos = dest.stat().st_size if dest.exists() else 0
            headers = dict(HEADERS)
            if pos:
                headers["Range"] = f"bytes={pos}-"

            with requests.get(url, headers=headers, stream=True, timeout=(10, 120)) as resp:
                if pos and resp.status_code == 200:  # el servidor ignoro el Range
                    pos, mode = 0, "wb"
                elif pos and resp.status_code == 206:
                    mode = "ab"
                else:
                    pos, mode = 0, "wb"

                total = int(resp.headers.get("Content-Length", 0)) + pos
                last_mark = -1
                with open(dest, mode) as fh:
                    for chunk in resp.iter_content(CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        pos += len(chunk)
                        if total:
                            pct = int(pos * 100 / total)
                            if pct // 5 != last_mark:
                                last_mark = pct // 5
                                print(f"    {dest.name}: {pct}% ({pos/1e6:.0f} MB)")

            if _md5(dest) == md5sum:
                return True
            print(f"    MD5 no coincide todavia; reintentando {dest.name}...")
        except (requests.RequestException, OSError) as exc:
            print(f"    Intento {attempt}/{retries} fallido ({exc}); reanudando en 3s...")
            time.sleep(3)

    return dest.exists() and _md5(dest) == md5sum


def download() -> None:
    cfg = load_config()
    raw_dir = ensure_dir(resolve(cfg["data"]["raw_dir"]))

    print("=" * 68)
    print("O.E.2 - DESCARGA DE DATOS (Sprint 1)")
    print("=" * 68)

    ok = True
    for split, (filename, md5sum) in SPLITS.items():
        dest = raw_dir / filename
        print(f"[{split}] {filename}")
        if _download_one(f"{BASE_URL}{filename}", dest, md5sum):
            print(f"  -> OK ({dest.stat().st_size/1e6:.1f} MB)")
        else:
            ok = False
            print(f"  -> FALLO (no se pudo completar {filename})")

    print("Descarga finalizada. Archivos .mat en 'data/raw'.")
    print("=" * 68)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    download()
