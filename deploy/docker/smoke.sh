#!/bin/bash
# 容器冒烟:起 CPU 容器 → https 200 → 无口令 403 → WS ready(真加载 whisper tiny/int8)
set -e
docker rm -f va-smoke 2>/dev/null || true
docker run -d --name va-smoke -p 18443:8443 \
  -e CAPTION_ACCESS_PW=smoketest \
  -e CAPTION_ASR_BACKEND=whisper -e CAPTION_ASR_DEVICE=cpu -e CAPTION_ASR_MODEL=tiny \
  -e CAPTION_LLM_URL= -e CAPTION_LLM_KEY= -e CAPTION_LLM_MODEL=x \
  va-web:cpu >/dev/null
for i in $(seq 1 60); do
  sleep 3
  C=$(curl -sk -o /dev/null -w %{http_code} --max-time 5 https://127.0.0.1:18443/ 2>/dev/null || true)
  [ "$C" = "200" ] && { echo "HTTP200_OK(第${i}次)"; break; }
  [ $i -eq 60 ] && { echo "❌ 容器没就绪"; docker logs va-smoke | tail -20; exit 1; }
done
S=$(curl -sk -o /dev/null -w %{http_code} --max-time 5 "https://127.0.0.1:18443/sessions" || true)
[ "$S" = "403" ] && echo "AUTH_403_OK" || echo "❌ 无口令返回了 $S"
docker exec -i va-smoke python3 - <<'PYEOF'
import asyncio, json, ssl, aiohttp
async def m():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
    async with aiohttp.ClientSession() as s:
        ws = await s.ws_connect("https://127.0.0.1:8443/ws?pw=smoketest", ssl=ctx)
        r = await ws.receive_json()
        assert r.get("status") == "ready", r
        frame = b"\x00\x00" * 480
        for _ in range(10): await ws.send_bytes(frame)
        await ws.send_str(json.dumps({"eof": 1})); await ws.close()
        print("WS_READY_OK")
asyncio.run(m())
PYEOF
docker logs va-smoke 2>&1 | grep -E "加载|就绪|自签" | head -3
docker rm -f va-smoke >/dev/null
echo "SMOKE_ALL_OK"
