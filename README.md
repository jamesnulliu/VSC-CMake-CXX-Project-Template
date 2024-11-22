# VSC-CMake-CXX-Project-Template
A Template of Cross-Platform CMake-C++ Project for Visual Studio Code with Github Actions CI/CD.

## 1. How to Use this Template

### 1.1. Generate a Project

Parameters:

- `<project-name>`: Name of your project. Can be any valid identifier.
- `<project-type>`: Type of the project you want to use. Can be:
  - `1`: cxx_exe, a simple C++ executable project.
  - `2`: cxx_lib, a C++ lib & test project.
  - `3`: cuda_exe, a CUDA executable project.
  - `4`: cuda_lib, a CUDA lib & test project.

```bash
python ./.templates/gen_project.py -n <project-name> -t <project-type>
```

### 1.2. Build exe/lib

After generation, build the project by:

```bash
# Linux
bash ./scripts/build.sh
# Windows
.\scripts\build.bat
```

### 1.3. Reset Project

If you chose a wrong template, don't be worry. Reset the project by:

```bash
python ./.templates/gen_project.py --reset
```

Or directly switch to a new template:

```bash
python ./.templates/gen_project.py -n <project-name> -t <project-type>
```

### 1.4. Remove Template

After generating your project, you can remove the ".template" directory by deleting it directly, or use the following command:

```bash
python ./.template/gen_project.py --remove-template
```
