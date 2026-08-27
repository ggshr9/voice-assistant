@echo off
REM voice-assistant 一键启动(Windows):Docker Desktop 起服务并打开浏览器。
REM 语音/大模型服务通过 deploy\docker\.env 指向你已有的 API。
cd /d "%~dp0..\deploy\docker"
where docker >nul 2>nul || (echo 请先安装 Docker Desktop: https://docs.docker.com/desktop/ & pause & exit /b 1)
if not exist .env (copy .env.example .env & echo 已生成 .env,请先编辑填入口令与 LLM 地址,然后重新运行 & notepad .env & exit /b 0)
docker compose --profile cpu up -d --build || (echo 启动失败,请看上方报错 & pause & exit /b 1)
echo 服务已启动,正在打开浏览器(自签证书警告点「高级-继续访问」即可)...
start https://localhost:8443/
