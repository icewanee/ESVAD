import requests
import json
import os
from multiprocessing import Pool, cpu_count
import shutil
import subprocess

OUTPUT_FOLDER = "deps"
DEPS_JSON = "deps.json"
MULTIPROCESSING_SIZE = min(cpu_count(), 16)


def download(d):
    print(d)
    url = f"https://samate.nist.gov/SARD/downloads/dependencies/{d}.zip"
    file = requests.get(url, stream=True)
    print(file.status_code)
    path = f"{OUTPUT_FOLDER}/{d}/{d}.zip"
    fold = f"{OUTPUT_FOLDER}/{d}"
    if os.path.exists(fold):
        shutil.rmtree(fold)
    os.mkdir(fold)
    with open(path, "wb") as f:
        f.write(file.raw.data)
    subprocess.run(["unzip", "-q", path, "-d", fold])
    os.remove(path)
    print(f"{d} finished")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.mkdir(OUTPUT_FOLDER)

    with open(DEPS_JSON, "r", encoding="utf-8") as f:
        deps = json.load(f)["deps"]
    with Pool(MULTIPROCESSING_SIZE) as p:
        res = p.map(download, deps)
