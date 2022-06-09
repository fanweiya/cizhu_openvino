for /F "TOKENS=2 " %%a in ('tasklist /FI "IMAGENAME eq rmq_main.exe" /nh') do taskkill /f /pid %%a
call openvino_deploy_package\bin\setupvars.bat
call cizhu_main.exe
