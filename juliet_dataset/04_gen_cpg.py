import os
import subprocess
import shutil
from multiprocessing import Pool
from tqdm import tqdm

PARALLEL_COMPILE_COUNT = 8  # can increase or decrease based on your cpu/ram
IN_DIR = "testcases_ir_normalized"
OUT_DIR = "testcases_cpg"
CPG_BIN_PATH = "/app/cpg-neo4j/bin/cpg-neo4j"


def gen_cpg(x):
    try:
        subprocess.run(
            x,
            check=True,
            env=dict(
                os.environ,
                **{
                    "JAVA_TOOL_OPTIONS": "-Dslf4j.provider=org.slf4j.helpers.NOP_FallbackServiceProvider -Dslf4j.internal.verbosity=WARN"
                },
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"found an error: {e}")


if __name__ == "__main__":
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.mkdir(OUT_DIR)

    to_cpg = []  # (in, out)
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
            bad_cpg = bad_ll[:-3] + ".json"
            to_cpg.append(
                (f"{cwe_in_bad_path}/{bad_ll}", f"{cwe_out_bad_path}/{bad_cpg}")
            )
        for good_ll in os.listdir(cwe_in_good_path):
            good_cpg = good_ll[:-3] + ".json"
            to_cpg.append(
                (f"{cwe_in_good_path}/{good_ll}", f"{cwe_out_good_path}/{good_cpg}")
            )

    commands = [
        [CPG_BIN_PATH, "--no-neo4j", "--export-json", out_path, in_path]
        for (in_path, out_path) in to_cpg
    ]

    with Pool(PARALLEL_COMPILE_COUNT) as p:
        results = list(tqdm(p.imap(gen_cpg, commands), total=len(commands)))
