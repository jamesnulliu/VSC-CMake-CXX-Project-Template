from argparse import ArgumentParser, ArgumentTypeError
import shutil
from pathlib import Path
import os
import atexit
import subprocess
import re

PROJECT_TYPE = {
    0: "cxx_exe",
    1: "cxx_lib",
    2: "cuda_exe",
    3: "cuda_lib",
}

TEMPLATE_DIR = Path(".templates")
GITHUB_WORKFLOWS_DIR = Path(".github/workflows")
LICENSE_PATH = Path("LICENSE")
README_PATH = Path("README.md")

TEMPLATE_GEN_PATHS = [
    "cmake",
    "include",
    "src",
    "lib",
    "test",
    "scripts",
    "CMakeLists.txt",
    ".clangd",
    ".clang-format",
    ".vscode",
    "vcpkg.json",
]

TARGET_EXTENSIONS = (
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cu",
    ".cuh",
    "CMakeLists.txt",
    ".cmake",
)

TEMPLATE_PROJECT_NAME = "_template_project_name_"


def parse_arguments():
    parser = ArgumentParser(description="Project type selector")
    parser.add_argument(
        "--remove-templates",
        action="store_true",
        help="Remove '.templates' and '.github/workflows'.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset template, remove all generated files",
    )
    parser.add_argument(
        "-t",
        "--project-type",
        type=int,
        choices=[t for t in range(len(PROJECT_TYPE))],
        help=f"Type of project to create: {PROJECT_TYPE}",
    )

    def validate_project_name(name: str):
        if not name.isidentifier():
            raise ArgumentTypeError(
                f"Project name '{name}' must be a valid identifier"
            )
        return name

    parser.add_argument(
        "-n",
        "--project-name",
        type=validate_project_name,
        help="Name of the project. Affects namespace and the include directory name.",
    )
    args = parser.parse_args()
    if not (args.reset or args.remove_templates) and (
        args.project_type is None or args.project_name is None
    ):
        parser.error(
            "--project-type and --project-name are required unless "
            "--remove-templates or --reset is specified"
        )

    return args


def get_cuda_arch():
    cmd = """
    if ! command -v nvcc &> /dev/null; then
        echo "-1"
        exit 1
    fi

    tmp_dir=$(mktemp -d)
    cat << 'EOF' > "${tmp_dir}/check_arch.cu"
#include <stdio.h>
int main() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("sm_%d%d", prop.major, prop.minor);
    return 0;
}
EOF

    if nvcc "${tmp_dir}/check_arch.cu" -o "${tmp_dir}/check_arch" &> \
        /dev/null && "${tmp_dir}/check_arch" 2>/dev/null; then
        rm -rf "$tmp_dir"
        exit 0
    else
        rm -rf "$tmp_dir"
        echo "-1"
        exit 1
    fi
    """
    result = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True
    )
    return result.stdout.strip()


def rcopy_files_from_src_dir_2_dst_dir(src_dir, dst_dir):
    """
    Recursively copy files from `src_dir` to `dst_dir`; The structure of the
    directory tree is maintained.
    """
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def replace_in_file(file_path: str, old: str, new: str):
    file_path = Path(file_path)
    with open(file_path, "r") as file:
        content = file.read()
    content = content.replace(old, new)
    with open(file_path, "w") as file:
        file.write(content)


def remove_line_in_file(file_path: str, pattern: str):
    file_path = Path(file_path)
    with open(file_path, "r") as file:
        lines = file.readlines()
    with open(file_path, "w") as file:
        for line in lines:
            if pattern not in line:
                file.write(line)


def get_linenos(file_path: str, pattern):
    file_path = Path(file_path)
    linenos = []
    with open(file_path, "r") as file:
        for lineno, line in enumerate(file.readlines()):
            if pattern in line:
                linenos.append(lineno)
    return linenos


def remove_lines_in_file(file_path: str, begin_lno: int, end_lno: int):
    with open(file_path, "r") as file:
        lines = file.readlines()
    end_lno = end_lno if end_lno > 0 else len(lines)
    lines = [
        l for lno, l in enumerate(lines) if lno < begin_lno or lno >= end_lno
    ]
    with open(file_path, "w") as file:
        for l in lines:
            file.write(l)


def replace_for_files_in_dir(directory, old, new):
    for root, _, files in Path(directory).walk():
        for file in files:
            if file.endswith(TARGET_EXTENSIONS):
                if file_path := Path(root, file):
                    replace_in_file(file_path, old, new)


class ProjectGenerator:
    def __init__(self, project_name: str, project_type: str):
        self.project_name = project_name
        self.project_type = PROJECT_TYPE[project_type]

    def run(self):
        shutil.copytree(TEMPLATE_DIR / "common", ".", dirs_exist_ok=True)
        shutil.copytree(
            TEMPLATE_DIR / self.project_type, ".", dirs_exist_ok=True
        )
        Path("./include", TEMPLATE_PROJECT_NAME).rename(
            Path("./include", self.project_name)
        )
        for directory in list(
            map(Path, ["include", "lib", "test", "src", "cmake"])
        ):
            replace_for_files_in_dir(
                directory,
                TEMPLATE_PROJECT_NAME + "_API",
                self.project_name.upper() + "_API",
            )
            replace_for_files_in_dir(
                directory,
                TEMPLATE_PROJECT_NAME + "_EXPORT",
                self.project_name.upper() + "_EXPORT",
            )
            replace_for_files_in_dir(
                directory,
                TEMPLATE_PROJECT_NAME + "_IMPORT",
                self.project_name.upper() + "_IMPORT",
            )
            replace_for_files_in_dir(
                directory, TEMPLATE_PROJECT_NAME, self.project_name
            )
        self.replace_cxx_api()
        replace_in_file(
            "CMakeLists.txt", TEMPLATE_PROJECT_NAME, self.project_name
        )
        # cuda exe or cuda lib
        if self.project_type.startswith("cuda"):
            if not (cuda_home := os.environ.get("CUDA_HOME", None)):
                cuda_home = os.environ.get("CUDA_DIR", None)
            if cuda_home is None:
                raise RuntimeError("CUDA_HOME or CUDA_DIR is not found")
            replace_in_file(".clangd", "<path-to-cuda>", cuda_home)
            replace_in_file(".clangd", "<cuda-arch>", get_cuda_arch())
        # c++ exe or c++ lib
        else:
            remove_lines_in_file(
                ".clangd", get_linenos(".clangd", "---")[-1], -1
            )
            remove_line_in_file(".clangd", "cuda")

    def replace_cxx_api(self):
        replace_for_files_in_dir(
            "include",
            f"{self.project_name}_API",
            f"{self.project_name.upper()}_API",
        )
        replace_for_files_in_dir(
            "include",
            f"{self.project_name}_EXPORT",
            f"{self.project_name.upper()}_EXPORT",
        )
        replace_for_files_in_dir(
            "include",
            f"{self.project_name}_IMPORT",
            f"{self.project_name.upper()}_IMPORT",
        )


def main(args):
    # Remove ".templates" directory on exit
    if args.remove_templates:
        to_remove = [
            TEMPLATE_DIR,
            GITHUB_WORKFLOWS_DIR,
            LICENSE_PATH,
            README_PATH,
        ]
        atexit.register(
            lambda: [
                shutil.rmtree(path, ignore_errors=True) for path in to_remove
            ]
        )
        return

    # Reset project directory by removing all template-generated file
    for path in TEMPLATE_GEN_PATHS:
        path = Path(path)
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    # Reset only
    if args.reset:
        shutil.copytree(TEMPLATE_DIR / "reset", ".", dirs_exist_ok=True)
        return

    generator = ProjectGenerator(args.project_name, args.project_type)
    generator.run()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
