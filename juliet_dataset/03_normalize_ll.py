import os
import subprocess
import shutil
from multiprocessing import Pool
from tqdm import tqdm

PARALLEL_COMPILE_COUNT = 8  # can increase or decrease based on your cpu/ram
IN_DIR = "testcases_ir"
OUT_DIR = "testcases_ir_normalized"


def normalize(x):
    try:
        subprocess.run(x, check=True)
    except Exception as e:
        print(f"found an error: {e}")


if __name__ == "__main__":
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.mkdir(OUT_DIR)

    to_normalize = []  # (in, out)
    for cwe_dir in os.listdir(IN_DIR):
        cwe_in_path = f"{IN_DIR}/{cwe_dir}"
        cwe_in_good_path = f"{cwe_in_path}/GOOD"
        cwe_in_bad_path = f"{cwe_in_path}/BAD"
        cwe_out_path = f"{OUT_DIR}/{cwe_dir}"
        cwe_out_good_path = f"{cwe_out_path}/GOOD"
        cwe_out_bad_path = f"{cwe_out_path}/BAD"

        os.mkdir(cwe_out_path)
        os.mkdir(cwe_out_bad_path)
        os.mkdir(cwe_out_good_path)

        for bad_ll in os.listdir(cwe_in_bad_path):
            to_normalize.append(
                (f"{cwe_in_bad_path}/{bad_ll}", f"{cwe_out_bad_path}/{bad_ll}")
            )
        for good_ll in os.listdir(cwe_in_good_path):
            to_normalize.append(
                (f"{cwe_in_good_path}/{good_ll}", f"{cwe_out_good_path}/{good_ll}")
            )

    commands = [
        [
            "opt-15",
            "-S",
            "-mem2reg",
            "-instcombine",
            "--strip-debug",
            "-o",
            out_path,
            in_path,
        ]
        for (in_path, out_path) in to_normalize
    ]

    with Pool(PARALLEL_COMPILE_COUNT) as p:
        results = list(tqdm(p.imap(normalize, commands), total=len(commands)))
