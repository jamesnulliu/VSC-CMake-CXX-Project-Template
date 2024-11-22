@echo off
setlocal

call scripts\setup-msbuild.bat

call bash .\scripts\build.sh %*

endlocal