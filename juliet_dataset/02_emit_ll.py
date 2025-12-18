import os
import subprocess
import shutil
from multiprocessing import Pool
from tqdm import tqdm

CODE_DIR = "testcases"
CODE_SUPPORT_DIR = "testcasesupport"
OUT_DIR = "testcases_ir"
PARALLEL_COMPILE_COUNT = 8  # can increase or decrease based on your cpu/ram
MINGW_VER = [x for x in os.listdir("/usr/lib/gcc/x86_64-w64-mingw32/") if "win32" in x][
    0
]


def process(inp):
    cwe_dir, ending_name, file_path = inp
    if file_path.endswith(".c"):
        # GOOD files
        out = subprocess.run(
            [
                "clang++-15",
                "-target",
                "x86_64-w64-windows-gnu",
                # 1. Force Windows behavior in Juliet headers
                "-D_WIN32",
                "-DUNICODE",
                "-D_UNICODE",
                "-D_CRT_SECURE_NO_WARNINGS",
                # 2. Block Linux header leakage & prioritize MinGW
                "-isystem",
                "/usr/x86_64-w64-mingw32/include",
                "-I",
                f"/usr/lib/gcc/x86_64-w64-mingw32/{MINGW_VER}/include/c++",
                "-I",
                f"/usr/lib/gcc/x86_64-w64-mingw32/{MINGW_VER}/include/c++/x86_64-w64-mingw32",
                "-S",
                "-O0",
                "-g",
                "-Xclang",
                "-no-opaque-pointers",
                "-Wno-everything",
                "-emit-llvm",
                "-I",
                CODE_SUPPORT_DIR,
                "-DINCLUDEMAIN",
                "-DOMITBAD",
                "-o",
                f"{cwe_dir}/GOOD/{ending_name}.good.ll",
                file_path,
            ],
            capture_output=True,
        )
        if out.returncode != 0:
            print(file_path, "Error", out.stderr)

        # BAD files
        out = subprocess.run(
            [
                "clang++-15",
                "-target",
                "x86_64-w64-windows-gnu",
                # 1. Force Windows behavior in Juliet headers
                "-D_WIN32",
                "-DUNICODE",
                "-D_UNICODE",
                "-D_CRT_SECURE_NO_WARNINGS",
                # 2. Block Linux header leakage & prioritize MinGW
                "-isystem",
                "/usr/x86_64-w64-mingw32/include",
                "-I",
                f"/usr/lib/gcc/x86_64-w64-mingw32/{MINGW_VER}/include/c++",
                "-I",
                f"/usr/lib/gcc/x86_64-w64-mingw32/{MINGW_VER}/include/c++/x86_64-w64-mingw32",
                "-S",
                "-O0",
                "-g",
                "-Xclang",
                "-no-opaque-pointers",
                "-Wno-everything",
                "-emit-llvm",
                "-I",
                CODE_SUPPORT_DIR,
                "-DINCLUDEMAIN",
                "-DOMITGOOD",
                "-o",
                f"{cwe_dir}/BAD/{ending_name}.bad.ll",
                file_path,
            ],
            capture_output=True,
        )
        if out.returncode != 0:
            print(file_path, "Error", out.stderr)

    elif file_path.endswith(".cpp"):
        # GOOD files
        out = subprocess.run(
            [
                "clang++-15",
                "-target",
                "x86_64-w64-windows-gnu",
                # 1. Force Windows behavior in Juliet headers
                "-D_WIN32",
                "-DUNICODE",
                "-D_UNICODE",
                "-D_CRT_SECURE_NO_WARNINGS",
                # 2. Block Linux header leakage & prioritize MinGW
                "-isystem",
                "/usr/x86_64-w64-mingw32/include",
                "-I",
                f"/usr/lib/gcc/x86_64-w64-mingw32/{MINGW_VER}/include/c++",
                "-I",
                f"/usr/lib/gcc/x86_64-w64-mingw32/{MINGW_VER}/include/c++/x86_64-w64-mingw32",
                "-S",
                "-O0",
                "-g",
                "-Xclang",
                "-no-opaque-pointers",
                "-Wno-everything",
                "-emit-llvm",
                "-I",
                CODE_SUPPORT_DIR,
                "-DINCLUDEMAIN",
                "-DOMITBAD",
                "-o",
                f"{cwe_dir}/GOOD/{ending_name}.good.ll",
                file_path,
            ],
            capture_output=True,
        )
        if out.returncode != 0:
            print(file_path, "Error", out.stderr)

        # BAD files
        out = subprocess.run(
            [
                "clang++-15",
                "-target",
                "x86_64-w64-windows-gnu",
                # 1. Force Windows behavior in Juliet headers
                "-D_WIN32",
                "-DUNICODE",
                "-D_UNICODE",
                "-D_CRT_SECURE_NO_WARNINGS",
                # 2. Block Linux header leakage & prioritize MinGW
                "-isystem",
                "/usr/x86_64-w64-mingw32/include",
                "-I",
                f"/usr/lib/gcc/x86_64-w64-mingw32/{MINGW_VER}/include/c++",
                "-I",
                f"/usr/lib/gcc/x86_64-w64-mingw32/{MINGW_VER}/include/c++/x86_64-w64-mingw32",
                "-S",
                "-O0",
                "-g",
                "-Xclang",
                "-no-opaque-pointers",
                "-Wno-everything",
                "-emit-llvm",
                "-I",
                CODE_SUPPORT_DIR,
                "-DINCLUDEMAIN",
                "-DOMITGOOD",
                "-o",
                f"{cwe_dir}/BAD/{ending_name}.bad.ll",
                file_path,
            ],
            capture_output=True,
        )
        if out.returncode != 0:
            print(file_path, "Error", out.stderr)


if __name__ == "__main__":
    print("Compiling LL")
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.mkdir(OUT_DIR)

    to_compile = []  # (cwe_folder, ending name, real filepath)
    for cwe_dir in os.listdir(CODE_DIR):
        code_path = f"{CODE_DIR}/{cwe_dir}"
        out_path = f"{OUT_DIR}/{cwe_dir}"
        os.mkdir(out_path)
        os.mkdir(f"{out_path}/GOOD")
        os.mkdir(f"{out_path}/BAD")

        for dir, wtf, files in os.walk(code_path):
            for file in files:
                file_path = f"{dir}/{file}"
                ending_name = file_path.replace(code_path, "").replace("/", "_")
                to_compile.append((out_path, ending_name, file_path))

    with Pool(PARALLEL_COMPILE_COUNT) as p:
        results = list(tqdm(p.imap(process, to_compile), total=len(to_compile)))
