from argparse import ArgumentParser, ArgumentTypeError
from enum import Enum, unique
import shutil
from pathlib import Path
import os


@unique
class ProjectType(Enum):
    cxx_exe = 1
    cxx_lib = 2
    cxx_cuda_exe = 3
    cxx_cuda_lib = 4


def copy_directory_contents(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # Create destination directory if it doesn't exist
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy each item in the source directory
    for item in src_dir.glob("*"):
        if item.is_file():
            shutil.copy2(item, dst_dir)
        elif item.is_dir():
            shutil.copytree(item, dst_dir / item.name, dirs_exist_ok=True)


def rename_directory(old_path, new_path):
    if Path(old_path).is_dir():
        Path(old_path).rename(new_path)


def replace_in_file(file_path, old_text, new_text):
    with open(file_path, "r") as file:
        content = file.read()

    content = content.replace(old_text, new_text)

    with open(file_path, "w") as file:
        file.write(content)


def replace_in_directory(directory, old_text, new_text):
    for root, _, files in Path(directory).walk():
        for file in files:
            if file.endswith((".cpp", ".hpp", ".h", ".cu", ".cuh")):
                file_path = Path(root, file)
                replace_in_file(
                    file_path,
                    f"{old_text}",
                    f"{new_text}",
                )


def main(args):
    paths_to_remove = [
        "cmake",
        "include",
        "src",
        "lib",
        "test",
        "scripts",
        "CMakeLists.txt",
        ".clangd",
        ".clang-format",
    ]

    for path in paths_to_remove:
        path = Path(path)
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    if args.remove_all:
        return

    template_dir = Path("templates")
    project_type = ProjectType(args.project_type).name
    match_pattern = "_template_project_name_"
    project_name = args.project_name

    copy_directory_contents(Path(template_dir, "common"), ".")
    copy_directory_contents(Path(template_dir, project_type), ".")

    rename_directory(Path("include", match_pattern), Path("include", project_name))

    for directory in ["include", "lib", "test", "src"]:
        if Path(directory).is_dir():
            replace_in_directory(directory, match_pattern, project_name)

    if Path("CMakeLists.txt").is_file():
        replace_in_file("CMakeLists.txt", match_pattern, project_name)

    if args.project_type in [3, 4]:
        if Path(".clangd").is_file():
            replace_in_file(".clangd", "/path/to/cuda", os.environ.get("CUDA_HOME", ""))


if __name__ == "__main__":
    parser = ArgumentParser(description="Project type selector")

    parser.add_argument(
        "-rma",
        "--remove-all",
        action="store_true",
        help="Remove generated files and directories and then exist.",
    )

    parser.add_argument(
        "-t",
        "--project-type",
        type=int,
        choices=[t.value for t in ProjectType],
        help="Type of project to create (1=CXX_EXE, 2=CXX_LIB, 3=CXX_CUDA_EXE, 4=CXX_CUDA_LIB)",
    )

    def validate_project_name(name):
        if not name.isidentifier():
            raise ArgumentTypeError(f"Project name '{name}' must be a valid identifier")
        return name

    parser.add_argument(
        "-n",
        "--project-name",
        type=validate_project_name,
        help="Name of the project. Affects namespace and the include directory name.",
    )

    args = parser.parse_args()

    if not args.remove_all and (args.project_type is None or args.project_name is None):
        parser.error(
            "--project-type and --project-name are required unless --remove-all is specified"
        )

    main(args)
