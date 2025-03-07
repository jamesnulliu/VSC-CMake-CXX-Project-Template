# VSC-CMake-CXX-Project-Template
A Template of Cross-Platform CMake-C++ Project for Visual Studio Code with Github Actions CI/CD.

## 1. How to Use this Template

### 1.1. Generate a Project

Parameters:

- `<project-name>`: Name of your project. Can be any valid identifier.
- `<project-type>`: Type of the project you want to use. Can be:
  - `0`: cxx_exe, a simple C++ executable project.
  - `1`: cxx_lib, a C++ lib & test project.
  - `2`: cuda_exe, a CUDA executable project.
  - `3`: cuda_lib, a CUDA lib & test project.

```bash
python ./.templates/gen_project.py -n <project-name> -t <project-type>
```

### 1.2. Build exe/lib

After generation, build the project by:

```bash
bash ./scripts/build.sh
```

Or if you are using windows and want to use MSVC deliberately, run the following command in Powershell or CMD:

```pwsh
.\scripts\msvc-bash.bat .\scripts\build.sh --prune-env-path
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
python ./.template/gen_project.py --remove-templates
```
