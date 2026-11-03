// End-to-end demo acceptance using installed Chrome and the Chrome DevTools Protocol.
// It only runs inference against an already-started local backend; it does not train.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const base = process.argv[2] || 'http://127.0.0.1:8774';
const port = 9227;
const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const profile = 'D:/Nam3_ki1/UFM-Rec_Backups/cleanup_20261029_nontraining/demo_ui_chrome_profile_20261029';
const output = path.join(root, 'reports/nontraining_20261029');
const downloads = path.join(output, 'demo_ui_downloads');
const screenshot = path.join(output, 'demo_ui_zero_shot_20261029.png');
const offlineScreenshot = path.join(output, 'demo_ui_offline_20261029.png');
const receiptPath = path.join(output, 'demo_ui_end_to_end_20261029.json');

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn, timeoutMs, label) {
  const started = Date.now();
  for (;;) {
    const value = await fn();
    if (value) return value;
    if (Date.now() - started > timeoutMs) throw new Error(`Timed out: ${label}`);
    await sleep(250);
  }
}

async function cdp(url) {
  const socket = new WebSocket(url);
  const pending = new Map();
  let next = 1;
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  socket.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (!msg.id || !pending.has(msg.id)) return;
    const {resolve, reject} = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? reject(new Error(`${msg.error.code}: ${msg.error.message}`)) : resolve(msg.result);
  };
  return {
    send(method, params={}) {
      return new Promise((resolve, reject) => {
        const id = next++;
        pending.set(id, {resolve, reject});
        socket.send(JSON.stringify({id, method, params}));
      });
    },
    close() { socket.close(); },
  };
}

async function main() {
  await fs.mkdir(profile, {recursive:true});
  await fs.mkdir(downloads, {recursive:true});
  assert.deepEqual(await fs.readdir(downloads), [], 'Download directory must be empty');
  const child = spawn(chrome, [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    '--disable-background-networking', '--disable-extensions', '--window-size=1440,1000',
    `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, 'about:blank',
  ], {windowsHide:true, stdio:'ignore'});
  let browser, tab;
  const receipt = {observed_at_utc:new Date().toISOString(), browser:'Chrome headless via CDP',
    backend_url:base, cases:[], status:'in_progress'};
  try {
    const version = await until(async () => {
      try { const r = await fetch(`http://127.0.0.1:${port}/json/version`); return r.ok ? r.json() : null; }
      catch { return null; }
    }, 20000, 'Chrome debugging endpoint');
    browser = await cdp(version.webSocketDebuggerUrl);
    const targetResponse = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method:'PUT'});
    assert.equal(targetResponse.ok, true);
    const target = await targetResponse.json();
    tab = await cdp(target.webSocketDebuggerUrl);
    await tab.send('Page.enable');
    await tab.send('Runtime.enable');
    await tab.send('Network.enable');
    await tab.send('Network.setBlockedURLs', {urls:['*m.media-amazon.com*']});
    await browser.send('Browser.setDownloadBehavior', {behavior:'allow', downloadPath:downloads, eventsEnabled:true});
    await tab.send('Page.navigate', {url:base});
    const evaluate = async expression => {
      const result = await tab.send('Runtime.evaluate', {expression, returnByValue:true, awaitPromise:true});
      if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
      return result.result.value;
    };
    await until(async () => await evaluate("document.querySelector('#connectionText')?.textContent === 'Demo sẵn sàng' && document.querySelector('#model')?.textContent === 'UFM Rec' && document.querySelectorAll('#matches .match').length > 0"), 60000, 'UFM UI ready');
    const status = await evaluate("({backend:document.querySelector('#model').textContent,catalog:document.querySelector('#catalogCount').textContent,matches:document.querySelectorAll('#matches .match').length})");
    assert.equal(status.backend, 'UFM Rec');
    assert.ok(status.catalog.includes('767'));
    receipt.cases.push({name:'ui_ready_real_backend', passed:true, observed:status});

    await evaluate("document.querySelector('#matches .match').click(); document.querySelector('#regime').value='zero_shot'; document.querySelector('#regime').dispatchEvent(new Event('change',{bubbles:true})); document.querySelector('#k').value='5'; document.querySelector('#k').dispatchEvent(new Event('change',{bubbles:true})); document.querySelector('#recommend').click(); true");
    await until(async () => await evaluate("!document.querySelector('#recommend').classList.contains('busy') && !document.querySelector('#export').disabled && document.querySelectorAll('#results .card').length === 5"), 120000, 'zero-shot results');
    const result = await evaluate("({history:document.querySelector('#historyCount').textContent,cards:document.querySelectorAll('#results .card').length,zeroShot:[...document.querySelectorAll('#results .card .badge')].every(e=>e.textContent==='Zero-shot'),imagesMissing:document.querySelectorAll('#results .product-visual.image-placeholder').length,exportEnabled:!document.querySelector('#export').disabled,model:lastResponse?.model,request:lastResponse?.request,results:lastResponse?.results})");
    assert.equal(result.history, '1/20');
    assert.equal(result.cards, 5);
    assert.equal(result.zeroShot, true);
    assert.equal(result.imagesMissing, 5);
    assert.equal(result.exportEnabled, true);
    assert.equal(result.request.regime, 'zero_shot');
    assert.equal(result.request.k, 5);
    assert.equal(result.results.length, 5);
    const shot = await tab.send('Page.captureScreenshot', {format:'png', captureBeyondViewport:true});
    await fs.writeFile(screenshot, Buffer.from(shot.data,'base64'));
    receipt.cases.push({name:'zero_shot_and_image_error',passed:true,history_asins:result.request.history,
      model_checkpoint_sha256:result.model.checkpoint_sha256, result_asins:result.results.map(x=>x.asin),
      missing_image_placeholders:result.imagesMissing, screenshot:path.relative(root,screenshot).replaceAll('\\','/')});

    await evaluate("document.querySelector('#export').click(); true");
    const downloaded = await until(async () => {
      const files = await fs.readdir(downloads);
      if (!files.includes('ufm-rec-demo.json')) return null;
      const p = path.join(downloads,'ufm-rec-demo.json');
      try {return JSON.parse(await fs.readFile(p,'utf8'));} catch {return null;}
    }, 20000, 'JSON download');
    assert.deepEqual(downloaded.request, result.request);
    assert.equal(downloaded.backend, 'UFM Rec');
    assert.equal(downloaded.model.checkpoint_sha256, result.model.checkpoint_sha256);
    assert.deepEqual(downloaded.results.map(x=>x.asin), result.results.map(x=>x.asin));
    assert.equal(downloaded.results.length, 5);
    receipt.cases.push({name:'downloaded_json_matches_ui',passed:true,file:path.relative(root,path.join(downloads,'ufm-rec-demo.json')).replaceAll('\\','/')});

    await tab.send('Network.emulateNetworkConditions', {offline:true, latency:0, downloadThroughput:0, uploadThroughput:0});
    await evaluate("document.querySelector('#recommend').click(); true");
    await until(async () => await evaluate("document.querySelector('#message').textContent.startsWith('Chưa tạo được gợi ý:') && !document.querySelector('#recommend').classList.contains('busy')"), 20000, 'offline failure state');
    const offline = await evaluate("({message:document.querySelector('#message').textContent,exportDisabled:document.querySelector('#export').disabled,resultsOpacity:document.querySelector('#results').style.opacity})");
    assert.equal(offline.exportDisabled, true);
    assert.equal(Number(offline.resultsOpacity), 0.55);
    const offlineShot = await tab.send('Page.captureScreenshot', {format:'png', captureBeyondViewport:true});
    await fs.writeFile(offlineScreenshot, Buffer.from(offlineShot.data,'base64'));
    receipt.cases.push({name:'offline_error_state',passed:true,observed:offline,screenshot:path.relative(root,offlineScreenshot).replaceAll('\\','/')});

    receipt.status='passed';
    receipt.summary='4/4 UI cases passed: real backend, zero-shot with blocked images, downloaded JSON, offline error state.';
  } catch (error) {
    receipt.status='failed';
    receipt.error=String(error.stack || error);
    throw error;
  } finally {
    await fs.writeFile(receiptPath, JSON.stringify(receipt,null,2)+'\n');
    try { await browser?.send('Browser.close'); } catch {}
    tab?.close(); browser?.close(); child.kill();
  }
  console.log(receipt.summary);
  console.log(receiptPath);
}
await main();
