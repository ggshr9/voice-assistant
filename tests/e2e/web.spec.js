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
    await page.goto(URL, { waitUntil: 'domcontentloaded' });
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
