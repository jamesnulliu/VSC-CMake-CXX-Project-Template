# Exit if not windows
if [ "$(uname -o)" != "Msys" ]; then
    echo "This script is only for Windows"
    exit 1
fi

batPath="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

cmd.exe /K "$batPath" x86_amd64