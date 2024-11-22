# VSC-CMake-CXX-Project-Template
A Template of Cross-Platform CMake-C++ Project for Visual Studio Code with Github Actions CI/CD.

## How to Use this Template

Parameters:

- `<project-name>`: Name of your project. Can be any string which is a valid identifier.
- `<project-type>`: Type of the project you want to use. Can be:
  - `1`: cxx_exe, a simple C++ executable project.
  - `2`: cxx_lib, a C++ lib & test project.
  - `3`: cuda_exe, a CUDA executable project.
  - `4`: cuda_lib, a CUDA lib & test project.

```bash
python templates/gen_project.py -n <project-name> -t <project-type>
```

After generation, build the project by:

```bash
# Linux
bash scripts/build.sh
# Windows
.\scripts\build.bat
```