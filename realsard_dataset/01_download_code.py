import requests
import json
import os
from multiprocessing import Pool, cpu_count
import shutil
from tqdm import tqdm
import zipfile

WANTED_CWES = [
    "CWE-78",
    "CWE-88",
    "CWE-89",
    "CWE-119",
    "CWE-120",
    "CWE-476",
    "CWE-682",
]
OUTPUT_FOLDER = "codes"
DEPS_OUTPUT = "deps.json"
MULTIPROCESSING_SIZE = min(cpu_count(), 16)


def get_uris(run):
    out = set()
    deps = tuple(sorted(run["properties"].get("dependencies", [])))
    for res in run["results"]:
        if "codeFlows" in res:
            for flow in res["codeFlows"]:
                for tflow in flow["threadFlows"]:
                    for loc in tflow["locations"]:
                        out.add(
                            (
                                loc["location"]["physicalLocation"]["artifactLocation"][
                                    "uri"
                                ],
                                res["ruleId"],
                                deps,
                            )
                        )
        else:
            for loc in res["locations"]:
                out.add(
                    (
                        loc["physicalLocation"]["artifactLocation"]["uri"],
                        res["ruleId"],
                        deps,
                    )
                )
    return out


def get_download_entries():
    with open("SARDMETA.json", "r") as f:
        data = json.load(f)["all"]

    input = []
    alldeps = set()
    for entry in data:
        assert len(entry["sarif"]["runs"]) == 1

        folder = entry["identifier"]
        for run in entry["sarif"]["runs"]:
            state = run["properties"]["state"]
            if state not in ["good", "bad"]:
                continue
            uris = get_uris(run)

            # unique pair of file and cwe in this run
            for uri in uris:
                _, cwe, deps = uri
                if cwe in WANTED_CWES:
                    alldeps.update(deps)
                    input.append((entry["download"], folder, f"{folder}.zip"))
    return input, alldeps


def download(tup):
    download, folder, name = tup
    path = f"{OUTPUT_FOLDER}/{folder}/{name}"
    fold = f"{OUTPUT_FOLDER}/{folder}"
    if os.path.exists(fold):
        shutil.rmtree(fold)
    os.mkdir(fold)
    file = requests.get(download, stream=True)
    if file.status_code == 200:
        try:
            with open(path, "wb") as fp:
                fp.write(file.raw.data)
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(fold)
            os.remove(path)
        except FileNotFoundError:
            print(f"For some reason file {name} is not found")
    else:
        shutil.rmtree(fold)
        print("FAILED", folder, name, file.status_code)
    return file.status_code


if __name__ == "__main__":
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.mkdir(OUTPUT_FOLDER)
    download_entries, deps = get_download_entries()
    with open(DEPS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"deps": list(deps)}, f)
    with Pool(MULTIPROCESSING_SIZE) as p:
        res = list(
            tqdm(p.imap(download, download_entries), total=len(download_entries))
        )
