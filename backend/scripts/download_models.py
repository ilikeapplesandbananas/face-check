import os
import sys
import zipfile
import shutil
import requests

def download_and_extract():
    cache_dir = os.path.expanduser("~/.insightface/models")
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "buffalo_l.zip")
    url = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip"

    existing_bytes = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
    print(f"Checking {zip_path}... current size: {existing_bytes} bytes")

    valid = False
    if existing_bytes > 0:
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                files = z.namelist()
                print("Existing zip is valid! Contains:", files)
                valid = True
        except Exception:
            print("Existing zip is partial, will resume...")

    if not valid:
        headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes > 0 else {}
        session = requests.Session()
        resp = session.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
        print(f"HTTP response code: {resp.status_code}")
        
        mode = "ab" if resp.status_code == 206 else "wb"
        with open(zip_path, mode) as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (5 * 1024 * 1024) < 1024 * 512:
                        print(f"Downloaded +{downloaded // (1024*1024)} MB...")

    extract_dir = os.path.join(cache_dir, "buffalo_l")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        print("Extracting files:", z.namelist())
        z.extractall(extract_dir)

    backend_models = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(backend_models, exist_ok=True)

    det_src = os.path.join(extract_dir, "det_10g.onnx")
    rec_src = os.path.join(extract_dir, "w600k_r50.onnx")

    det_dest = os.path.join(backend_models, "retinaface_r50_v1.onnx")
    rec_dest = os.path.join(backend_models, "arcface_r100_v1.onnx")

    shutil.copyfile(det_src, det_dest)
    shutil.copyfile(rec_src, rec_dest)

    print("SUCCESS: Models copied to:")
    print(" - Detector:", det_dest, f"({os.path.getsize(det_dest):,} bytes)")
    print(" - Recognizer:", rec_dest, f"({os.path.getsize(rec_dest):,} bytes)")

if __name__ == "__main__":
    download_and_extract()
