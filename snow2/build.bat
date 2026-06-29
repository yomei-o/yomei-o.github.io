@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /utf-8 /O2 /EHsc /std:c++17 snowflake_kobayashi.cpp /Fe:snowflake_kobayashi.exe
