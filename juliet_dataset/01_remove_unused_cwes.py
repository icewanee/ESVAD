import shutil
import os

TESTCASES_PATH = "testcases"
WANTED_CWES = set(
    [
        "CWE78",
        "CWE90",
        "CWE122",
        "CWE134",
        "CWE190",
        "CWE401",
        "CWE415",
        "CWE590",
        "CWE690",
        "CWE762",
        "CWE789",
    ]
)

if __name__ == "__main__":
    for dir in os.listdir(TESTCASES_PATH):
        cwe_name = dir.split("_")[0]
        if cwe_name not in WANTED_CWES:
            shutil.rmtree(f"{TESTCASES_PATH}/{dir}")
    print("Removed unused CWEs")
