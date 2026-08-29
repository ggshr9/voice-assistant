// 网页版 UI 的端到端测试（Playwright）。
//
// 为什么值得有：web/ 这半边是【对外服务】,带访问口令,而此前全部覆盖都是纯逻辑单测 ——
// 口令门、上传、会议库这些真正面向用户的路径一次都没被自动验证过。
// 实测就靠手点才发现上传功能已经坏了两个月。
//
// 需要两个环境变量（都不写进仓库：那是你的网络拓扑和口令）：
//   CAPTION_URL=https://127.0.0.1:21443   建 ssh 隧道后指本地即可
//   CAPTION_PW=xxxx
// 跑: npx playwright test tests/e2e/web.spec.js
import { test, expect } from '@playwright/test';

const URL = process.env.CAPTION_URL;
const PW = process.env.CAPTION_PW || '';

test.skip(!URL, 'CAPTION_URL 未设置');
test.use({ ignoreHTTPSErrors: true });   // 隧道上证书域名对不上是正常的

async function enter(page) {
  await page.goto(URL);
  await page.evaluate(() => { try { sessionStorage.clear(); } catch (e) {} });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.fill('#pw', PW);
  await page.click('#enter');
  await expect(page.locator('#gate')).toBeHidden();
}

test.describe('口令门', () => {
  test('未输入口令时挡在门外,且没有任何会议数据泄漏', async ({ page }) => {
    // #gate 是 position:fixed;inset:0 的覆盖层,背后的 #homeView 在 DOM 里仍算
    // "visible" —— 所以断言可见性没有意义。真正要紧的是:鉴权之前不该有数据。
    // 监听器里绝不能抛异常 —— 会卡住导航(踩过:new URL() 遇到非 http 协议就炸)
    const fetched = [];
    page.on('request', r => {
      const u = r.url();
      if (u.includes('/session') || u.endsWith('/sessions')) fetched.push(u.slice(-40));
    });
    // 必须自己保证起点干净:前端「sessionStorage 里有口令就自动进门」是**正确行为**,
    // 而同一 worker 里前面的测试会留下口令 —— 这条于是随机挂,而产品并没有回归。
    // (实测:全量跑挂、单独跑过 —— 典型的测试间污染。)
    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { try { sessionStorage.clear(); } catch (e) {} });
    fetched.length = 0;                       // 清掉刚才那次加载产生的请求记录
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.locator('#gate')).toBeVisible();
    await page.waitForTimeout(2000);
    expect(fetched, `鉴权前就请求了：${fetched.join(', ')}`).toEqual([]);
    expect(await page.locator('#lib > *').count()).toBe(0);
  });

  test('门是全屏覆盖的,点不到背后的控件', async ({ page }) => {
    await page.goto(URL);
    const box = await page.locator('#gate').boundingBox();
    const vp = page.viewportSize();
    expect(box.width).toBeGreaterThanOrEqual(vp.width - 1);
    expect(box.height).toBeGreaterThanOrEqual(vp.height - 1);
  });

  test('空口令点进入不放行', async ({ page }) => {
    await page.goto(URL);
    await page.click('#enter');
    await expect(page.locator('#gate')).toBeVisible();
  });

  test('错误口令会被服务端打回并重新弹门', async ({ page }) => {
    await page.goto(URL);
    await page.fill('#pw', '肯定不对的口令');
    await page.click('#enter');
    await expect(page.locator('#gate')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('#gerr')).not.toBeEmpty();
  });

  test('正确口令进入并加载会议库', async ({ page }) => {
    await enter(page);
    await expect(page.locator('#homeView')).toBeVisible();
  });

  test('刷新后不用重输(sessionStorage)', async ({ page }) => {
    await enter(page);
    await page.reload();
    await expect(page.locator('#gate')).toBeHidden();
  });
});

test.describe('会议库', () => {
  test.beforeEach(async ({ page }) => await enter(page));

  test('列出已有会议', async ({ page }) => {
    await page.waitForTimeout(1500);
    const n = await page.locator('#lib > *').count();
    expect(n).toBeGreaterThan(0);
  });

  test('搜索能过滤,清空能还原', async ({ page }) => {
    await page.waitForTimeout(1500);
    const all = await page.locator('#lib > *').count();
    await page.fill('#libSearch', '绝不可能匹配的字符串xyzzy');
    await page.waitForTimeout(400);
    expect(await page.locator('#lib > *').count()).toBeLessThan(all);
    await page.fill('#libSearch', '');
    await page.waitForTimeout(400);
    expect(await page.locator('#lib > *').count()).toBe(all);
  });

  test('刷新按钮不报错', async ({ page }) => {
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await page.click('#refreshBtn');
    await page.waitForTimeout(1500);
    expect(errs).toEqual([]);
  });
});

test.describe('页面健康度', () => {
  test('控制台没有报错', async ({ page }) => {
    const errs = [];
    page.on('console', m => m.type() === 'error' && errs.push(m.text()));
    page.on('pageerror', e => errs.push(e.message));
    await enter(page);
    await page.waitForTimeout(2500);
    expect(errs, `控制台报错：${errs.join(' | ')}`).toEqual([]);
  });

  test('窄屏下不出现横向滚动', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await enter(page);
    const overflow = await page.evaluate(() =>
      document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow, '页面在手机宽度下横向溢出').toBeLessThanOrEqual(1);
  });
});

test.describe('上传 → 纪要（这条路曾坏了两个月）', () => {
  // 需要 CAPTION_FIXTURE 指向一段短音频；没有就跳过
  const FIXTURE = process.env.CAPTION_FIXTURE;

  test('上传后能拿到 job 并最终产出纪要', async ({ page }) => {
    test.skip(!FIXTURE, 'CAPTION_FIXTURE 未设置');
    test.setTimeout(10 * 60 * 1000);
    await enter(page);
    await page.fill('#title', 'E2E自动化_' + Date.now());
    await page.setInputFiles('#upfile', FIXTURE);
    // 上传成功会跳进详情页；失败则 alert
    const alerts = [];
    page.on('dialog', async d => { alerts.push(d.message()); await d.dismiss(); });
    await expect(page.locator('#detailView')).toBeVisible({ timeout: 120000 });
    expect(alerts, `上传被拒：${alerts.join(' | ')}`).toEqual([]);
    // 等纪要出来（含分人+ASR+LLM，几分钟）
    await expect(page.locator('#minutesCard')).toBeVisible({ timeout: 9 * 60 * 1000 });
    await expect(page.locator('#md')).toContainText(/摘要|决议|待办/);
  });

  test('口令不对时上传不该在服务器留下文件', async ({ page }) => {
    test.skip(!FIXTURE, 'CAPTION_FIXTURE 未设置');
    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    // 绕过前端直接打接口，模拟脱离 UI 的攻击者
    const status = await page.evaluate(async (u) => {
      const fd = new FormData();
      fd.append('pw', '肯定不对');
      fd.append('audio', new File([new Uint8Array(1024)], 'x.m4a'));
      const r = await fetch(u + '/session/upload', { method: 'POST', body: fd });
      return r.status;
    }, URL.replace(/\/$/, ''));
    expect(status, '口令错误必须被拒').toBe(403);
  });
});

test.describe('会话生命周期：改名 / 删除', () => {
  const FIXTURE = process.env.CAPTION_FIXTURE;
  test.skip(!FIXTURE, 'CAPTION_FIXTURE 未设置');

  // 建一场临时会话。标题一律带 E2E_ 前缀，afterEach 靠它兜底清理 ——
  // 只靠「测试自己删」不够：断言一挂就直接抛出，删除那步永远走不到，
  // 服务器上就攒下一堆垃圾会话（实测第一次跑就留了 3 个）。
  async function makeSession(page, title) {
    await enter(page);
    await page.fill('#title', title);
    await page.setInputFiles('#upfile', FIXTURE);
    await expect(page.locator('#detailView')).toBeVisible({ timeout: 120000 });
    return await page.evaluate(() => window.curSid || null);
  }

  test.afterEach(async ({ page }) => {
    // 用接口直接删，不依赖 UI ——  UI 可能正停在失败时的任意状态
    await page.evaluate(async (pw) => {
      const r = await fetch('/sessions?pw=' + encodeURIComponent(pw));
      const d = await r.json().catch(() => ({}));
      for (const s of (d.sessions || d || [])) {
        if (typeof s?.title === 'string' && s.title.startsWith('E2E')) {
          await fetch('/session/' + encodeURIComponent(s.id) + '/delete?pw='
                      + encodeURIComponent(pw), { method: 'POST' }).catch(() => {});
        }
      }
    }, PW);
  });

  test('改名后标题即时更新，刷新后仍在', async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    const orig = 'E2E改名前_' + Date.now();
    await makeSession(page, orig);
    await expect(page.locator('#dTitle')).toHaveText(orig);

    const renamed = orig.replace('改名前', '改名后');
    page.once('dialog', d => d.accept(renamed));      // prompt()
    await page.getByRole('button', { name: /改名/ }).click();
    await expect(page.locator('#dTitle')).toHaveText(renamed);

    // 回首页再进来，确认是真落盘了而不是只改了 DOM
    await page.click('#homeBtn');
    await page.waitForTimeout(1200);
    await expect(page.locator('#lib')).toContainText(renamed);
  });

  test('改名传空标题不生效（不该把标题清掉）', async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    const orig = 'E2E空标题_' + Date.now();
    await makeSession(page, orig);
    page.once('dialog', d => d.accept('   '));        // 只有空白
    await page.getByRole('button', { name: /改名/ }).click();
    await page.waitForTimeout(800);
    await expect(page.locator('#dTitle')).toHaveText(orig, { timeout: 5000 });
  });

  test('取消改名对话框不改动任何东西', async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    const orig = 'E2E取消_' + Date.now();
    await makeSession(page, orig);
    page.once('dialog', d => d.dismiss());
    await page.getByRole('button', { name: /改名/ }).click();
    await page.waitForTimeout(800);
    await expect(page.locator('#dTitle')).toHaveText(orig);
  });

  test('删除后回到首页且列表里不再出现', async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    const title = 'E2E待删_' + Date.now();
    await makeSession(page, title);
    page.once('dialog', d => d.accept());             // confirm()
    await page.getByRole('button', { name: /删除/ }).click();
    await expect(page.locator('#homeView')).toBeVisible({ timeout: 20000 });
    await page.waitForTimeout(1500);
    await expect(page.locator('#lib')).not.toContainText(title);
  });

  test('取消删除对话框则会话仍在', async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    const title = 'E2E不删_' + Date.now();
    await makeSession(page, title);
    page.once('dialog', d => d.dismiss());
    await page.getByRole('button', { name: /删除/ }).click();
    await page.waitForTimeout(1000);
    await expect(page.locator('#dTitle')).toHaveText(title);
    // 收尾：真的删掉，别给服务器留垃圾
    page.once('dialog', d => d.accept());
    await page.getByRole('button', { name: /删除/ }).click();
    await expect(page.locator('#homeView')).toBeVisible({ timeout: 20000 });
  });
});

test.describe('接口层安全（绕过前端直接打）', () => {
  async function api(page, path, init) {
    return await page.evaluate(async ([u, i]) => {
      const r = await fetch(u, i);
      let body = null;
      try { body = await r.json(); } catch (e) { /* 非 JSON */ }
      return { status: r.status, body };
    }, [path, init || {}]);
  }

  test('删除接口的 sid 不能穿越目录', async ({ page }) => {
    await enter(page);
    for (const evil of ['../', '..', '../../etc', '%2e%2e%2f', '/etc']) {
      const r = await api(page,
        `/session/${encodeURIComponent(evil)}/delete?pw=${encodeURIComponent(PW)}`,
        { method: 'POST' });
      expect(r.status, `${evil} 没被拦住`).not.toBe(200);
    }
  });

  test('所有写接口都要口令', async ({ page }) => {
    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    const paths = [
      ['/session/x/delete', 'POST'],
      ['/session/x/rename', 'POST'],
      ['/session/x/minutes', 'POST'],
      ['/sessions', 'GET'],
    ];
    for (const [p, method] of paths) {
      const r = await api(page, p, { method });
      expect([401, 403], `${p} 无口令时返回了 ${r.status}`).toContain(r.status);
    }
  });
});

test.describe('断线重连时不丢音频', () => {
  // 这几条不需要真麦克风：直接在页面里驱动 sendPcm/flushPcmQueue 的状态机。
  // 真实的丢失场景是「重连退避最长 5 秒，一次抖动就是 5 秒话没了」，
  // 而用户只看到「断线重连中」，不会知道内容缺了一块。
  async function setup(page) {
    await enter(page);
    await page.evaluate(() => {
      // 页面脚本顶层的 let 进的是全局【词法】环境,不是 window 的属性 ——
      // 写 window.ws 影响不到 sendPcm 真正读的那个绑定(第一版就栽在这里)。
      window.__sent = [];
      capCh = 1; capturing = true; manualStop = false; retry = 1;
      pcmQueue = []; pcmQueuedBytes = 0; pcmDroppedMs = 0;
      ws = { readyState: 3, send: b => window.__sent.push(b.byteLength) };  // 3 = CLOSED
    });
  }
  const chunk = 'new ArrayBuffer(3200)';   // 0.1 秒 @16k 单声道 s16

  test('断线期间的音频被缓存而不是丢弃', async ({ page }) => {
    await setup(page);
    const r = await page.evaluate(([c]) => {
      for (let i = 0; i < 10; i++) sendPcm(eval(c));
      return { sent: window.__sent.length, queued: pcmQueue.length, dropped: pcmDroppedMs };
    }, [chunk]);
    expect(r.sent, '断线时不该发出去').toBe(0);
    expect(r.queued, '断线期间的音频被丢弃了').toBe(10);
    expect(r.dropped, '没超上限却报了丢失').toBe(0);
  });

  test('重连后缓存的音频全部补发', async ({ page }) => {
    await setup(page);
    const r = await page.evaluate(([c]) => {
      for (let i = 0; i < 10; i++) sendPcm(eval(c));
      ws.readyState = 1;                        // 连上了
      flushPcmQueue();
      return { sent: window.__sent.length, queued: pcmQueue.length };
    }, [chunk]);
    expect(r.sent, '补发的块数不对').toBe(10);
    expect(r.queued).toBe(0);
  });

  test('补发保持原顺序', async ({ page }) => {
    await setup(page);
    const ok = await page.evaluate(() => {
      window.__sent = [];
      ws.send = b => window.__sent.push(new Uint8Array(b)[0]);
      for (let i = 1; i <= 5; i++) { const b = new ArrayBuffer(3200); new Uint8Array(b)[0] = i; sendPcm(b); }
      ws.readyState = 1; flushPcmQueue();
      return JSON.stringify(window.__sent);
    });
    expect(ok, '补发顺序乱了 —— 音频错序等于内容错乱').toBe('[1,2,3,4,5]');
  });

  test('超过上限时丢最旧的并明确报出丢了多久', async ({ page }) => {
    await setup(page);
    const r = await page.evaluate(([c]) => {
      // 上限 90s；灌 120s 进去（每块 0.1s）
      for (let i = 0; i < 1200; i++) sendPcm(eval(c));
      return { queuedSec: Math.round(pcmQueuedBytes / (16000 * 2)), droppedSec: Math.round(pcmDroppedMs / 1000) };
    }, [chunk]);
    expect(r.queuedSec, '缓存没被限制住，会吃爆内存').toBeLessThanOrEqual(90);
    expect(r.droppedSec, '丢了却没记下来').toBeGreaterThan(0);
    expect(r.queuedSec + r.droppedSec).toBeGreaterThanOrEqual(119);   // 总量对得上
  });

  test('丢失时状态栏明确告知，不能静默', async ({ page }) => {
    await setup(page);
    const txt = await page.evaluate(([c]) => {
      for (let i = 0; i < 1200; i++) sendPcm(eval(c));
      return document.querySelector('#st')?.textContent || '';
    }, [chunk]);
    expect(txt, `状态栏没提丢失：${txt}`).toMatch(/丢/);
  });

  test('连接正常时不经过队列', async ({ page }) => {
    await setup(page);
    const r = await page.evaluate(([c]) => {
      ws.readyState = 1;
      for (let i = 0; i < 5; i++) sendPcm(eval(c));
      return { sent: window.__sent.length, queued: pcmQueue.length };
    }, [chunk]);
    expect(r.sent).toBe(5);
    expect(r.queued, '正常时不该攒队列').toBe(0);
  });
});

test.describe('麦克风选错时要说话', () => {
  // 真实踩到的坑：录了 19 秒、音频确实传到服务器（opus 10KB）、状态显示「录制中」，
  // 唯独全是静音 —— 因为页面从不指定 deviceId，Chrome 按站点记忆挑到了虚拟声卡
  // （BlackHole / 聚合设备在没有音频路由进去时输出纯零）。界面完全没提示。
  async function armMeter(page, rms) {
    await enter(page);
    return await page.evaluate((rms) => {
      capturing = true;
      micStream = { getAudioTracks: () => [{ label: 'BlackHole 2ch' }] };
      // 造一个恒定 rms 的假 analyser
      const an = {
        fftSize: 256,
        getFloatTimeDomainData: b => { for (let i = 0; i < b.length; i++) b[i] = rms; },
      };
      startMeter(an);
      return true;
    }, rms);
  }

  test('正常有声时显示设备名，不报警', async ({ page }) => {
    await armMeter(page, 0.05);
    await page.waitForTimeout(600);
    const el = page.locator('#micName');
    await expect(el).toContainText('BlackHole 2ch');
    await expect(el).not.toHaveClass(/bad/);
  });

  test('持续零电平超过阈值就明确报警', async ({ page }) => {
    await armMeter(page, 0);
    // 把静音起点往前拨，免得真等 12 秒
    await page.evaluate(() => { silentSince = Date.now() - 13000; });
    await page.waitForTimeout(600);
    const el = page.locator('#micName');
    await expect(el).toHaveClass(/bad/);
    await expect(el).toContainText('没有声音');
    await expect(el, '要告诉用户去哪儿改').toContainText(/换设备|地址栏/);
  });

  test('报警里带上是哪个设备', async ({ page }) => {
    await armMeter(page, 0);
    await page.evaluate(() => { silentSince = Date.now() - 13000; });
    await page.waitForTimeout(600);
    await expect(page.locator('#micName'), '不说是哪个设备，用户不知道该换掉什么')
      .toContainText('BlackHole 2ch');
  });

  test('声音恢复后报警自动撤掉', async ({ page }) => {
    await armMeter(page, 0);
    await page.evaluate(() => { silentSince = Date.now() - 13000; });
    await page.waitForTimeout(500);
    await expect(page.locator('#micName')).toHaveClass(/bad/);
    await page.evaluate(() => {
      // 换成有声的 analyser
      cancelAnimationFrame(meterRaf);
      startMeter({ fftSize: 256, getFloatTimeDomainData: b => { for (let i = 0; i < b.length; i++) b[i] = 0.05; } });
    });
    await page.waitForTimeout(600);
    await expect(page.locator('#micName')).not.toHaveClass(/bad/);
  });

  test('安静房间的底噪不算静音', async ({ page }) => {
    // -60dB 左右：真人不说话时的环境底噪，不该被当成设备故障
    await armMeter(page, 0.001);
    await page.evaluate(() => { silentSince = Date.now() - 13000; });
    await page.waitForTimeout(600);
    await expect(page.locator('#micName'), '把正常的安静误报成故障，警告就没人信了')
      .not.toHaveClass(/bad/);
  });
});

test.describe('麦克风选择与开录前试音', () => {
  // 一场两小时的会录成空的是不可恢复的。事后警告救不回来，
  // 所以要在【开录之前】就能选设备、能试音，并且开录时自动挡一道。

  test('设备下拉存在且默认是「系统默认」', async ({ page }) => {
    await enter(page);
    await expect(page.locator('#micSel')).toBeVisible();
    await expect(page.locator('#micSel')).toHaveValue('');
  });

  test('选择被记住（换页面也还在）', async ({ page }) => {
    await enter(page);
    await page.evaluate(() => {
      const sel = document.getElementById('micSel');
      sel.innerHTML = '<option value="">系统默认</option><option value="dev-abc">某麦克风</option>';
      sel.value = 'dev-abc';
      sel.dispatchEvent(new Event('change'));
    });
    expect(await page.evaluate(() => localStorage.getItem('cap_mic_id'))).toBe('dev-abc');
    await page.reload();
    expect(await page.evaluate(() => savedMicId()), '刷新后选择丢了').toBe('dev-abc');
  });

  test('选了设备就必须把 deviceId 传下去', async ({ page }) => {
    await enter(page);
    const c = await page.evaluate(() => {
      localStorage.setItem('cap_mic_id', 'dev-xyz');
      return JSON.stringify(micConstraints());
    });
    expect(c, 'deviceId 没传下去，选了等于没选').toContain('dev-xyz');
    expect(c, '必须 exact，否则浏览器会"尽量满足"然后悄悄换一个').toContain('exact');
  });

  test('没选时不带 deviceId（用系统默认）', async ({ page }) => {
    await enter(page);
    const c = await page.evaluate(() => {
      localStorage.removeItem('cap_mic_id');
      return JSON.stringify(micConstraints());
    });
    expect(c).not.toContain('deviceId');
    expect(c, '别把降噪等基础约束弄丢').toContain('echoCancellation');
  });

  test('设备被拔掉后清掉失效的记忆', async ({ page }) => {
    await enter(page);
    const left = await page.evaluate(async () => {
      localStorage.setItem('cap_mic_id', 'dev-已拔掉');
      navigator.mediaDevices.enumerateDevices = async () =>
        [{ kind: 'audioinput', deviceId: 'dev-still-here', label: '还在的麦克风' }];
      await loadMicList();
      return localStorage.getItem('cap_mic_id');
    });
    expect(left, '记着一个不存在的设备，下次 exact 约束会直接失败').toBeNull();
  });

  test('probeLevel 听到声音时返回非零峰值', async ({ page }) => {
    await enter(page);
    const peak = await page.evaluate(async () => {
      const ctx = new AudioContext();
      const osc = ctx.createOscillator(); osc.frequency.value = 440;
      const dst = ctx.createMediaStreamDestination();
      osc.connect(dst); osc.start();
      const p = await probeLevel(dst.stream, 400);
      osc.stop(); ctx.close();
      return p;
    });
    expect(peak, '有声音却探不到，开录就会误报').toBeGreaterThan(0.0008);
  });

  test('probeLevel 对静音流返回接近零', async ({ page }) => {
    await enter(page);
    const peak = await page.evaluate(async () => {
      const ctx = new AudioContext();
      const dst = ctx.createMediaStreamDestination();   // 什么都不接 = 纯静音
      const p = await probeLevel(dst.stream, 400);
      ctx.close();
      return p;
    });
    expect(peak, '静音没被识别出来，那道防线就是摆设').toBeLessThan(0.0008);
  });

  test('probeLevel 出错时放行而不是堵死流程', async ({ page }) => {
    await enter(page);
    const p = await page.evaluate(async () => await probeLevel(null, 200));
    expect(p, '探测失败应放行 —— 宁可放行也别把能用的流程堵死').toBeGreaterThan(0.0008);
  });
});

test.describe('手记 × 转写合并（Granola 式）', () => {
  const FIXTURE = process.env.CAPTION_FIXTURE;
  test.skip(!FIXTURE, 'CAPTION_FIXTURE 未设置');

  test.afterEach(async ({ page }) => {
    await page.evaluate(async (pw) => {
      const r = await fetch('/sessions?pw=' + encodeURIComponent(pw)).then(r => r.json()).catch(() => ({}));
      for (const s of (r.sessions || r || []))
        if (typeof s?.title === 'string' && s.title.startsWith('E2E'))
          await fetch('/session/' + encodeURIComponent(s.id) + '/delete?pw=' + encodeURIComponent(pw), { method: 'POST' }).catch(() => {});
    }, PW);
  });

  test('notes 接口无口令被拒且不泄漏会话存在性', async ({ page }) => {
    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    const r = await page.evaluate(async () => {
      const a = await fetch('/session/真实不存在的id/notes', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: 'x' }) });
      return a.status;
    });
    expect(r, '无口令必须 403（404 会泄漏会话是否存在）').toBe(403);
  });

  test('手记保存 → 详情页可见可编辑 → 重新生成出增强笔记', async ({ page }) => {
    test.setTimeout(10 * 60 * 1000);
    // 1) 上传建会话（自动出纪要）
    await enter(page);
    await page.fill('#title', 'E2E手记_' + Date.now());
    await page.setInputFiles('#upfile', FIXTURE);
    await expect(page.locator('#detailView')).toBeVisible({ timeout: 120000 });
    await expect(page.locator('#minutesCard')).toBeVisible({ timeout: 9 * 60 * 1000 });
    const sid = await page.evaluate(() => curSid);

    // 2) 会后补写手记（走接口，等价于详情页编辑）
    const note = '- 缓存定了用 Redis\n- 压测口径待确认XYZZY';
    const ok = await page.evaluate(async ([sid, pw, note]) => {
      const r = await fetch('/session/' + encodeURIComponent(sid) + '/notes', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pw, text: note }) });
      return (await r.json()).ok === true;
    }, [sid, PW, note]);
    expect(ok, '手记保存失败').toBe(true);

    // 3) 刷新详情:手记卡可见且内容在
    await page.evaluate((sid) => openDetail(sid), sid);
    await expect(page.locator('#notesCard')).toBeVisible({ timeout: 20000 });
    await expect(page.locator('#dNotesTa')).toHaveValue(/XYZZY/);

    // 4) 重新生成 → 增强笔记卡出现,且保留了用户原话（骨架不许改写）
    await page.getByRole('button', { name: /重新生成/ }).click();
    await expect(page.locator('#enhancedCard')).toBeVisible({ timeout: 9 * 60 * 1000 });
    await expect(page.locator('#ed'), '用户手记的原话必须保留在增强笔记里')
      .toContainText('XYZZY', { timeout: 30000 });
  });
});

test.describe('跨会议检索问答', () => {
  test('无口令打 /ask 被拒', async ({ page }) => {
    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    const st = await page.evaluate(async () => {
      const r = await fetch('/ask', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ q: 'x' }) });
      return r.status;
    });
    expect(st).toBe(403);
  });

  test('空问题不发请求', async ({ page }) => {
    await enter(page);
    const sent = [];
    page.on('request', r => { if (r.url().includes('/ask')) sent.push(1); });
    await page.click('#askBtn');
    await page.waitForTimeout(500);
    expect(sent.length).toBe(0);
  });

  test('真实问答:出答案与出处,出处可点进详情', async ({ page }) => {
    test.setTimeout(4 * 60 * 1000);
    await enter(page);
    await page.fill('#askQ', '最近有哪些会议？简单说说各自谈了什么');
    await page.click('#askBtn');
    await expect(page.locator('#askCard')).toBeVisible();
    // LLM 两段式,给足时间;答案区最终要有实质内容
    await expect(page.locator('#askAns')).not.toBeEmpty({ timeout: 3 * 60 * 1000 });
    const txt = await page.locator('#askAns').textContent();
    expect(txt.length, '答案太短不像真的答了').toBeGreaterThan(20);
    const chips = page.locator('#askSrc .srcchip');
    await expect(chips.first(), '没有出处 chip').toBeVisible();
    await chips.first().click();
    await expect(page.locator('#detailView')).toBeVisible({ timeout: 15000 });
  });
});

test.describe('暂停/继续（issue #1）', () => {
  // 暂停的语义是**没录**:接电话是隐私场景,那几分钟的音频一个字节都不该离开本机。
  async function armed(page) {
    await enter(page);
    await page.evaluate(() => {
      window.__sent = [];
      capturing = true; manualStop = false; retry = 0; capCh = 1;
      paused = false; pausedTotal = 0; pausedAt = 0;
      pcmQueue = []; pcmQueuedBytes = 0; pcmDroppedMs = 0;
      $('pauseBtn').disabled = false;
      ws = { readyState: 1, send: b => window.__sent.push(b.byteLength) };
    });
  }
  const push = `(buf => { if(!paused) sendPcm(buf); })(new ArrayBuffer(3200))`;

  test('暂停期间音频被丢弃——不发送也绝不进缓冲队列', async ({ page }) => {
    await armed(page);
    const r = await page.evaluate(([push]) => {
      setPaused(true);
      window.__sent = [];                       // 清掉 pause 控制帧
      for (let i = 0; i < 10; i++) eval(push);
      return { sent: window.__sent.length, queued: pcmQueue.length };
    }, [push]);
    expect(r.sent, '暂停期间发出去了').toBe(0);
    expect(r.queued, '暂停期间被攒进了缓冲——那等于还是录了').toBe(0);
  });

  test('暂停/恢复各发一个控制帧(不再伪造静音)', async ({ page }) => {
    // 第一版靠塞 1 秒零 PCM 冲句 —— 每次暂停录音里多 1 秒无中生有的音频,
    // 且服务端不知道暂停发生过,音频/墙钟两条时间轴对不上。
    // 现在的契约:pause/resume 各一个 JSON 文本帧,定稿与 pause_spans 由服务端做。
    await armed(page);
    const r = await page.evaluate(() => {
      const texts = [], bins = [];
      ws.send = m => (typeof m === 'string' ? texts : bins).push(m);
      setPaused(true); setPaused(false);
      return { texts, bins: bins.length };
    });
    expect(r.bins, '不该再发伪造音频帧').toBe(0);
    expect(r.texts.length).toBe(2);
    expect(JSON.parse(r.texts[0]).pause).toBe(1);
    expect(JSON.parse(r.texts[1]).resume).toBe(1);
  });

  test('恢复后继续发送', async ({ page }) => {
    await armed(page);
    const r = await page.evaluate(([push]) => {
      setPaused(true); setPaused(false);
      window.__sent = [];
      for (let i = 0; i < 5; i++) eval(push);
      return window.__sent.length;
    }, [push]);
    expect(r).toBe(5);
  });

  test('状态栏明示暂停,恢复后回到录制中', async ({ page }) => {
    await armed(page);
    await page.evaluate(() => setPaused(true));
    await expect(page.locator('#st')).toContainText('暂停');
    await expect(page.locator('#st'), '要说清后果').toContainText('不会被录制');
    await page.evaluate(() => setPaused(false));
    await expect(page.locator('#st')).toContainText('录制中');
  });

  test('暂停不触发麦克风静音误报', async ({ page }) => {
    await armed(page);
    await page.evaluate(() => {
      micStream = { getAudioTracks: () => [{ label: '内建麦克风' }] };
      setPaused(true);
      startMeter({ fftSize: 256, getFloatTimeDomainData: b => b.fill(0) });
      silentSince = Date.now() - 20000;          // 伪造已静音 20 秒
    });
    await page.waitForTimeout(600);
    await expect(page.locator('#micName'), '暂停时零电平是预期,不该报设备故障')
      .not.toHaveClass(/bad/);
  });

  test('未开录时暂停按钮不可用', async ({ page }) => {
    await enter(page);
    await expect(page.locator('#pauseBtn')).toBeDisabled();
  });
});

test.describe('停止即自动出最终质量', () => {
  test('录满 15 秒停止后自动触发生成', async ({ page }) => {
    await enter(page);
    const fired = await page.evaluate(async () => {
      // 只驱动 stopAll 的决策逻辑:采集/WS 全用桩
      let started = null;
      window.startMinutes = id => { started = id; };
      window.stopAudio = () => {}; window.waitWsClosed = async () => {};
      window.flushNotes = async () => {};
      capturing = true; manualStop = false; curSid = 'sid-测试';
      t0 = Date.now() - 20000; pausedTotal = 0; pausedAt = 0;   // 录了 20 秒
      ws = { readyState: 3, send: () => {}, close: () => {} };
      await stopAll();
      return started;
    });
    expect(fired, '停止后没有自动生成').toBe('sid-测试');
  });

  test('太短的场不烧 GPU', async ({ page }) => {
    await enter(page);
    const fired = await page.evaluate(async () => {
      let started = null;
      window.startMinutes = id => { started = id; };
      window.stopAudio = () => {}; window.waitWsClosed = async () => {};
      window.flushNotes = async () => {};
      capturing = true; manualStop = false; curSid = 'sid-短';
      t0 = Date.now() - 5000; pausedTotal = 0; pausedAt = 0;    // 只录了 5 秒
      ws = { readyState: 3, send: () => {}, close: () => {} };
      await stopAll();
      return started;
    });
    expect(fired, '5 秒的误触也去生成纪要 = 浪费').toBeNull();
  });

  test('暂停时间不计入录制时长', async ({ page }) => {
    await enter(page);
    const sec = await page.evaluate(() => {
      t0 = Date.now() - 60000; pausedTotal = 50000; pausedAt = 0;  // 60s 里暂停了 50s
      return recordedSec();
    });
    expect(sec, '暂停的 50 秒不该算录制').toBeLessThan(15);
  });
});

test.describe('素材质检横幅', () => {
  test('音乐场详情页出警告横幅(真实会话)', async ({ page }) => {
    await enter(page);
    await page.evaluate(() => openDetail('20260827_1921_会议_c20e'));
    const b = page.locator('#matBanner');
    await expect(b).toBeVisible({ timeout: 20000 });
    await expect(b).toContainText('人声');
    await expect(b, '要给出可操作的解释,不是干巴巴一个百分比').toContainText('背景音乐');
  });

  test('正常会话不弹横幅', async ({ page }) => {
    await enter(page);
    // 找一场没有 speech_ratio 或占比正常的旧会
    const sid = await page.evaluate(async (pw) => {
      const r = await fetch('/sessions?pw=' + encodeURIComponent(pw)).then(r => r.json());
      const list = r.sessions || r || [];
      const ok = list.find(s => s.id !== '20260827_1921_会议_c20e');
      return ok && ok.id;
    }, PW);
    test.skip(!sid, '没有其他会话可对照');
    await page.evaluate(id => openDetail(id), sid);
    await page.waitForTimeout(2500);
    await expect(page.locator('#matBanner')).toHaveCount(0);
  });
});
