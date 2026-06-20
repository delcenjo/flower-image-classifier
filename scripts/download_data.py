import tarfile
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
DEST_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    archive = DEST_DIR / "flower_photos.tgz"
    print("Downloading dataset ...")
    urllib.request.urlretrieve(URL, archive)
    print("Extracting ...")
    with tarfile.open(archive) as tar:
        tar.extractall(DEST_DIR)
    archive.unlink()
    print(f"Done -> {DEST_DIR / 'flower_photos'}")


if __name__ == "__main__":
    main()
