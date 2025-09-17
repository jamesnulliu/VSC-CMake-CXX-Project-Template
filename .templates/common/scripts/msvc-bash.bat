@echo off
setlocal

call scripts\msvc-setup.bat

@REM Suppose that bash is in the PATH
call bash %*

endlocal