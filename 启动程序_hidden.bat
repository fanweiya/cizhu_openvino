@echo off
if "%1" == "h" goto begin
mshta vbscript:createobject("wscript.shell").run("""%~nx0"" h",0)(window.close)&&exit
:begin
REM
for /F "TOKENS=2 " %%a in ('tasklist /FI "IMAGENAME eq cizhu_main.exe" /nh') do taskkill /f /pid %%a
call openvino_deploy_package\bin\setupvars.bat
call cizhu_main.exe
