"""Ham UCI dosyasından (zip veya txt) SQLite veritabanını kurar. Kullanım: python scripts/build_db.py [--src dosya]"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energyradar.config import DATA_DIR, DB_PATH  # noqa: E402
from energyradar.data import build_db  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="zip veya txt (varsayılan: data/uci içinde bulunan)")
    ap.add_argument("--db", default=str(DB_PATH))
    a = ap.parse_args()
    src = a.src
    if not src:
        import zipfile
        found = []
        for z in sorted(DATA_DIR.glob("*.zip")):
            try:
                if any(n.endswith("household_power_consumption.txt") for n in zipfile.ZipFile(z).namelist()):
                    found.append(z)
                else:
                    print(f"Atlandı (UCI verisi değil): {z.name}")
            except zipfile.BadZipFile:
                print(f"Atlandı (bozuk zip): {z.name}")
        found += sorted(DATA_DIR.glob("household_power_consumption.txt"))
        if not found:
            sys.exit(f"{DATA_DIR} içinde UCI verisi yok. archive.ics.uci.edu/dataset/235 adresinden indirdiğiniz zip'i oraya koyun "
                     "(proje zip'ini değil).")
        src = found[0]
    print(f"Kaynak: {src}\nVeritabanı kuruluyor (yaklaşık 20-60 sn)...")
    rep = build_db(src, a.db)
    print("Tamam:", rep)


if __name__ == "__main__":
    main()
