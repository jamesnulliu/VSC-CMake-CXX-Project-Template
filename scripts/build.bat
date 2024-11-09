@echo off

call scripts\setup-msbuild.bat

call bash .\scripts\build.sh %*