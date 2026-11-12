// CheckBit — AI anomaly detection dashboard (collector status, model output,
// detection history, evidence, user feedback, admin controls).
// Shows only what the API actually returns: no dummy prices or scores.
(() => {
  "use strict";
  const API = "/api";
  const $ = (id) => document.getElementById(id);
  const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const won = (n) => (Number.isFinite(+n) ? "₩" + Math.round(+n).toLocaleString("ko-KR") : "—");
  const kst = (iso, withDate = false) => {
    const d = new Date(iso);
    if (isNaN(d)) return "—";
    return new Intl.DateTimeFormat("ko-KR", { timeZone: "Asia/Seoul", hour: "2-digit", minute: "2-digit", hour12: false,
      ...(withDate ? { month: "numeric", day: "numeric" } : {}) }).format(d);
  };
  const ago = (iso) => {
    if (!iso) return "—";
    const s = Math.round((Date.now() - new Date(iso)) / 1000);
    return s < 60 ? `${s}초 전` : s < 3600 ? `${Math.round(s / 60)}분 전` : `${Math.round(s / 3600)}시간 전`;
  };
  const LABELS = {
    log_return: "1분 수익률", hl_range: "1분 고저 변동폭", rv_10: "10분 실현 변동성", d_log_volume: "거래량 변화",
    log_value: "거래대금 규모", d_log_value: "거래대금 변화", log_trade_count: "체결 건수",
    buy_sell_imbalance: "매수·매도 체결 불균형", spread_bps: "호가 스프레드", l1_imbalance: "최우선 호가 잔량 불균형",
    depth_imbalance: "전체 호가 잔량 불균형",
  };
  function fmtFeature(name, v) {
    if (v === null || v === undefined || !Number.isFinite(+v)) return "—";
    switch (name) {
      case "log_return": return `${(Math.expm1(v) * 100 >= 0 ? "+" : "")}${(Math.expm1(v) * 100).toFixed(3)}%`;
      case "hl_range": case "rv_10": return `${(v * 100).toFixed(3)}%`;
      case "d_log_volume": case "d_log_value": return `${Math.exp(v).toFixed(2)}배`;
      case "log_value": return `${(Math.expm1(v) / 100).toFixed(2)}억 원`;
      case "log_trade_count": return `${Math.round(Math.expm1(v))}건`;
      case "spread_bps": return `${(+v).toFixed(2)}bp`;
      default: return `${v >= 0 ? "+" : ""}${(+v).toFixed(2)}`;
    }
  }
  async function getJSON(path, opts = {}) {
    const res = await fetch(API + path, { cache: "no-store", headers: { Accept: "application/json", ...(opts.headers || {}) }, ...opts });
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON error page */ }
    if (!res.ok) throw Object.assign(new Error((body && body.detail) || `HTTP ${res.status}`), { status: res.status });
    return body;
  }
  const state = { range: 180, feature: "__value", timeline: null, selected: null, aiStatus: null };

  function notice(text, kind = "warn") {
    const el = $("aiNotice");
    if (!text) { el.hidden = true; return; }
    el.hidden = false; el.dataset.kind = kind; el.textContent = text;
  }

  // ------------------------------------------------------------ status cards
  function renderCollector(st) {
    const led = $("colLed");
    if (!st) { led.dataset.state = "bad"; $("colState").textContent = "API 오류"; $("colDetail").textContent = "수집 상태를 불러오지 못했습니다."; return; }
    const rows = Number(st.rows || 0).toLocaleString("ko-KR");
    if (st.running && st.receiving) { led.dataset.state = "ok"; $("colState").textContent = "수집 중 · 연결됨"; }
    else if (st.running) { led.dataset.state = "warn"; $("colState").textContent = st.connected ? "연결됨 · 수신 대기" : "재연결 중"; }
    else { led.dataset.state = "bad"; $("colState").textContent = "수집 중지"; }
    $("colDetail").textContent = `누적 ${rows}건 · 마지막 체결 ${ago(st.last_trade_at)}${st.reconnects ? ` · 재연결 ${st.reconnects}회` : ""}`;
  }
  function renderModel(ai) {
    const led = $("modelLed");
    if (!ai) { led.dataset.state = "bad"; $("modelState").textContent = "API 오류"; $("modelDetail").textContent = "모델 상태를 불러오지 못했습니다."; return; }
    if (!ai.model_loaded) {
      led.dataset.state = "bad"; $("modelState").textContent = "모델 준비 안 됨";
      $("modelDetail").textContent = ai.model_error || "학습된 모델이 없습니다.";
      return;
    }
    const m = ai.model;
    led.dataset.state = ai.worker_alive ? "ok" : "warn";
    $("modelState").textContent = `LSTM-AE · 임계값 ${m.threshold.toFixed(3)}`;
    const label = m.data_label === "synthetic" ? "⚠ 합성 데이터로 학습한 시험용 모델" : "실제 수집 데이터 학습";
    $("modelDetail").textContent = `${m.version} · ${label} · 윈도우 ${m.window}분 · ${ai.ready ? "1분마다 실시간 분석" : `분석 준비 ${ai.ready_bars}/${ai.required_bars}분`}`;
  }
  function renderJudge(latest) {
    const card = $("judgeCard");
    if (!latest || latest.status !== "ok") {
      card.dataset.status = "unknown";
      if (latest && latest.status === "warming_up") {
        $("judgeBadge").textContent = "준비";
        $("judgeState").textContent = "실시간 분석 준비 중";
        $("judgeDetail").textContent = `연속 1분 데이터 ${latest.ready_bars}/${latest.required_bars}분 · 이후 1분마다 자동 분석`;
        return;
      }
      $("judgeBadge").textContent = "–";
      $("judgeState").textContent = latest && latest.status === "model_not_ready" ? "모델 미실행" : "판정 대기";
      $("judgeDetail").textContent = (latest && latest.status === "model_not_ready" && latest.message) || "분석 결과를 불러오는 중입니다.";
      return;
    }
    card.dataset.status = latest.is_anomaly ? "anomaly" : "normal";
    $("judgeBadge").textContent = latest.is_anomaly ? "이상" : "정상";
    $("judgeState").textContent = latest.is_anomaly ? "이상 징후 탐지" : "정상 패턴";
    $("judgeDetail").textContent = `${kst(latest.detected_at, true)} KST · 점수 ${latest.score.toFixed(3)} / 임계값 ${latest.threshold.toFixed(3)}`;
  }
  function renderHistory(ai, fb) {
    if (!ai || !ai.store) { $("histState").textContent = "—"; return; }
    $("histState").textContent = `이상 ${Number(ai.store.anomalies).toLocaleString("ko-KR")}건`;
    const agree = fb && fb.agreement_rate !== null && fb.agreement_rate !== undefined ? ` · AI-사용자 일치 ${(fb.agreement_rate * 100).toFixed(0)}%` : "";
    $("histDetail").textContent = `전체 분석 ${Number(ai.store.analyses).toLocaleString("ko-KR")}회 · 사용자 검토 ${fb ? fb.reviewed : 0}건${agree}`;
  }

  // ------------------------------------------------------------------ chart
  function featureValue(p) {
    if (state.feature === "__value") return p.value;
    const v = p.features ? p.features[state.feature] : null;
    return v === null || v === undefined ? null : v;
  }
  function drawChart() {
    const svg = $("aiChart"), tl = state.timeline, empty = $("aiChartEmpty");
    const pts = tl ? tl.points : [];
    if (pts.length < 2) {
      svg.innerHTML = ""; svg._c = null; empty.style.display = "grid";
      empty.textContent = tl && tl.message ? tl.message : "표시할 수집 데이터가 없습니다. 데이터 수집을 시작하세요.";
      return;
    }
    empty.style.display = "none";
    const mobile = $("aiChartWrap").clientWidth < 530;
    const w = mobile ? 375 : 960, h = mobile ? 380 : 430, L = mobile ? 6 : 12, R = mobile ? 58 : 82, xmax = w - R, pw = xmax - L;
    const panels = [{ key: "price", top: 20, h: mobile ? 112 : 132 }, { key: "score", top: mobile ? 162 : 184, h: mobile ? 92 : 104 },
      { key: "feat", top: mobile ? 284 : 320, h: mobile ? 60 : 74 }];
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    const n = pts.length, x = (i) => L + (i / (n - 1)) * pw, bw = Math.max(1.5, pw / n);
    const th = tl.threshold;
    const scale = (vals, p, padFrac = 0.1, includeZero = false) => {
      const f = vals.filter(Number.isFinite);
      if (!f.length) return null;
      let lo = Math.min(...f), hi = Math.max(...f);
      if (includeZero) lo = Math.min(lo, 0);
      if (hi === lo) { hi += Math.abs(hi) * 0.001 + 1e-9; lo -= Math.abs(lo) * 0.001 + 1e-9; }
      const pad = (hi - lo) * padFrac; lo -= includeZero && lo === 0 ? 0 : pad; hi += pad;
      return { lo, hi, y: (v) => p.top + ((hi - v) / (hi - lo)) * p.h };
    };
    const sp = scale(pts.map((p) => p.close), panels[0]);
    const ss = scale(pts.map((p) => p.score).concat(Number.isFinite(th) ? [th] : []), panels[1], 0.08, true);
    const fv = pts.map(featureValue);
    const sf = scale(fv, panels[2], 0.1, state.feature === "__value");
    const path = (vals, y) => { let d = "", pen = false; vals.forEach((v, i) => { if (!Number.isFinite(v)) { pen = false; return; } d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1) + " "; pen = true; }); return d; };
    const grid = (p, s, fmt) => !s ? "" : [0, 0.5, 1].map((f) => { const yy = p.top + p.h * f, v = s.hi - (s.hi - s.lo) * f; return `<line x1="${L}" x2="${xmax}" y1="${yy}" y2="${yy}" stroke="#eaf0ec" stroke-dasharray="4 5"/><text x="${w - 2}" y="${yy + 4}" text-anchor="end" fill="#99aaa1" font-size="11">${esc(fmt(v))}</text>`; }).join("");
    const ptitle = (p, t) => `<text x="${L + 2}" y="${p.top - 7}" fill="#6f8679" font-size="11" font-weight="700">${esc(t)}</text>`;
    let out = "";
    // Anomaly bands across all panels (status color + icon below; never color alone).
    pts.forEach((p, i) => { if (p.is_anomaly) out += `<rect x="${(x(i) - bw / 2).toFixed(1)}" y="${panels[0].top}" width="${bw.toFixed(1)}" height="${panels[2].top + panels[2].h - panels[0].top}" fill="#e9565a" fill-opacity=".09"/>`; });
    out += grid(panels[0], sp, (v) => (v / 1e6).toFixed(2) + "M") + grid(panels[1], ss, (v) => v.toFixed(2));
    out += grid(panels[2], sf, (v) => state.feature === "__value" ? (v / 1e8).toFixed(1) + "억" : fmtFeature(state.feature, v));
    out += ptitle(panels[0], "수집 가격 (KRW)") + ptitle(panels[1], "이상 점수") + ptitle(panels[2], state.feature === "__value" ? "1분 거래대금" : LABELS[state.feature]);
    if (sp) out += `<path d="${path(pts.map((p) => p.close), sp.y)}" fill="none" stroke="#138a5b" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    if (ss) {
      if (Number.isFinite(th)) out += `<line x1="${L}" x2="${xmax}" y1="${ss.y(th)}" y2="${ss.y(th)}" stroke="#d08a2c" stroke-width="1.5" stroke-dasharray="6 4"/><text x="${xmax - 4}" y="${ss.y(th) - 5}" text-anchor="end" fill="#a86d1c" font-size="10" font-weight="700">임계값 ${th.toFixed(3)}</text>`;
      out += `<path d="${path(pts.map((p) => p.score), ss.y)}" fill="none" stroke="#4f6bd8" stroke-width="2" stroke-linejoin="round"/>`;
    } else {
      out += `<text x="${(L + xmax) / 2}" y="${panels[1].top + panels[1].h / 2}" text-anchor="middle" fill="#9aaba2" font-size="12">${esc(tl.model_version ? "실시간 분석 준비 중" : "모델이 준비되지 않아 이상 점수를 계산하지 않았습니다")}</text>`;
    }
    if (sf) {
      if (state.feature === "__value") pts.forEach((p, i) => { if (Number.isFinite(p.value) && p.value > 0) { const y0 = sf.y(0), y1 = sf.y(p.value); out += `<rect x="${(x(i) - bw * 0.4).toFixed(1)}" y="${y1.toFixed(1)}" width="${(bw * 0.8).toFixed(1)}" height="${Math.max(0.5, y0 - y1).toFixed(1)}" fill="#9fcbb4" rx="${Math.min(2, bw * 0.3).toFixed(1)}"/>`; } });
      else out += `<path d="${path(fv, sf.y)}" fill="none" stroke="#5c7a6b" stroke-width="1.6"/>`;
    }
    pts.forEach((p, i) => {
      if (!p.is_anomaly) return;
      if (sp) out += `<circle cx="${x(i)}" cy="${sp.y(p.close)}" r="4.5" fill="#e9565a" stroke="#fff" stroke-width="2"/>`;
      if (ss && Number.isFinite(p.score)) out += `<circle cx="${x(i)}" cy="${ss.y(p.score)}" r="4.5" fill="#e9565a" stroke="#fff" stroke-width="2"/>`;
    });
    const ticks = (mobile ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).map((f) => { const i = Math.round(f * (n - 1)); return `<text x="${x(i)}" y="${h - 4}" text-anchor="${f === 0 ? "start" : f === 1 ? "end" : "middle"}" fill="#9aaba1" font-size="11">${esc(kst(pts[i].t, state.range > 360))}</text>`; }).join("");
    out += ticks + `<line id="aiHover" x1="0" x2="0" y1="${panels[0].top}" y2="${panels[2].top + panels[2].h}" stroke="#477263" stroke-dasharray="3 4" visibility="hidden"/>`;
    svg.innerHTML = out;
    svg._c = { pts, x, L, xmax, w, n };
  }
  function hoverIndex(e) {
    const c = $("aiChart")._c; if (!c) return null;
    const b = $("aiChart").getBoundingClientRect(); if (!b.width) return null;
    const cx = ((e.clientX - b.left) / b.width) * c.w;
    if (cx < c.L - 6 || cx > c.xmax + 6) return null;
    return Math.max(0, Math.min(c.n - 1, Math.round(((cx - c.L) / (c.xmax - c.L)) * (c.n - 1))));
  }
  function chartMove(e) {
    const i = hoverIndex(e), c = $("aiChart")._c, tip = $("aiTooltip"), line = $("aiHover");
    if (i === null || !c) { chartLeave(); return; }
    const p = c.pts[i];
    line.setAttribute("x1", c.x(i)); line.setAttribute("x2", c.x(i)); line.setAttribute("visibility", "visible");
    const judge = p.score === null ? "분석 없음" : p.is_anomaly ? "🔴 이상 판정" : "🟢 정상";
    const fname = state.feature === "__value" ? "거래대금" : LABELS[state.feature];
    const fval = state.feature === "__value" ? (p.value / 1e8).toFixed(2) + "억 원" : fmtFeature(state.feature, featureValue(p));
    tip.innerHTML = `<div class="label">${esc(kst(p.t, true))} KST${p.filled ? " · 체결 없음(보간)" : ""}</div><b>${won(p.close)}</b>` +
      `<div class="label">점수: ${p.score === null ? "—" : p.score.toFixed(3)}${Number.isFinite(state.timeline.threshold) ? " / 임계 " + state.timeline.threshold.toFixed(3) : ""}</div>` +
      `<div class="label">${esc(fname)}: ${esc(fval)}</div><div class="label">${judge}${p.analysis_id ? " · 클릭하여 상세" : ""}</div>`;
    const b = $("aiChartWrap").getBoundingClientRect();
    tip.style.display = "block";
    tip.style.left = Math.max(90, Math.min(b.width - 90, e.clientX - b.left)) + "px";
    tip.style.top = Math.max(110, e.clientY - b.top - 10) + "px";
  }
  function chartLeave() { $("aiTooltip").style.display = "none"; const l = $("aiHover"); if (l) l.setAttribute("visibility", "hidden"); }
  function chartClick(e) {
    const i = hoverIndex(e), c = $("aiChart")._c;
    if (i === null || !c) return;
    // pick the nearest analysed minute (anomalies preferred) within ±2 points
    let best = null;
    for (let k = Math.max(0, i - 2); k <= Math.min(c.n - 1, i + 2); k++) {
      const p = c.pts[k]; if (!p.analysis_id) continue;
      const rank = (p.is_anomaly ? 0 : 10) + Math.abs(k - i);
      if (!best || rank < best.rank) best = { rank, id: p.analysis_id };
    }
    if (best) loadDetail(best.id);
  }

  // -------------------------------------------------------------- history
  async function loadDetections() {
    const body = $("detBody");
    try {
      const d = await getJSON("/ai/detections?limit=50&anomalies_only=true");
      $("detCount").textContent = d.model_version ? `현재 모델 기준 최근 ${d.count}건` : "모델 미준비";
      body.replaceChildren();
      if (!d.items.length) {
        body.innerHTML = `<tr><td colspan="4" class="ai-empty">${d.model_version ? "아직 이상으로 판정된 구간이 없습니다." : "모델이 준비되지 않아 탐지 이력이 없습니다."}</td></tr>`;
        return;
      }
      // Group consecutive anomalous minutes (gap <= 2 min) into one anomaly interval.
      const groups = [];
      for (const it of d.items) { // newest first
        const g = groups[groups.length - 1];
        if (g && g.startMinute - it.minute <= 2) { g.items.push(it); g.startMinute = it.minute; if (it.score > g.peak.score) g.peak = it; }
        else groups.push({ items: [it], startMinute: it.minute, endMinute: it.minute, peak: it, start: it, end: it });
      }
      $("detCount").textContent += ` · ${groups.length}개 구간`;
      for (const g of groups) {
        const it = g.peak, first = g.items[g.items.length - 1], last = g.items[0];
        const tr = document.createElement("tr");
        tr.tabIndex = 0; tr.dataset.id = it.id;
        if (state.selected === it.id) tr.setAttribute("aria-selected", "true");
        const top = (it.top_features || []).slice(0, 2).map((t) => `${t.label} ${t.contribution_pct}%`).join(", ");
        const labels = g.items.map((x) => x.user_label).filter(Boolean);
        const ul = it.user_label || labels[0];
        const lab = ul ? { normal: "정상", anomaly: "이상", hold: "보류" }[ul] : "미검토";
        const span = g.items.length > 1 ? `${esc(kst(first.detected_at, true))}~${esc(kst(last.detected_at))}<br><span class="micro">${g.items.length}분 · 최고점 ${esc(kst(it.detected_at))}</span>` : esc(kst(it.detected_at, true));
        tr.innerHTML = `<td>${span}</td><td><b>${it.score.toFixed(3)}</b> / ${it.threshold.toFixed(3)}<br><span class="ai-pill" data-v="anomaly">×${(it.score / it.threshold).toFixed(2)}</span></td><td>${esc(top || "—")}</td><td><span class="ai-pill" data-v="${esc(ul || "")}">${esc(lab)}</span></td>`;
        tr.addEventListener("click", () => loadDetail(it.id));
        tr.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); loadDetail(it.id); } });
        body.append(tr);
      }
    } catch (err) {
      body.innerHTML = `<tr><td colspan="4" class="ai-empty">탐지 이력을 불러오지 못했습니다 (${esc(err.message)}).</td></tr>`;
    }
  }

  async function loadDetail(id) {
    state.selected = id;
    try { history.replaceState(null, "", `#detection-${id}`); } catch { /* ignore */ }
    document.querySelectorAll("#detBody tr").forEach((tr) => tr.setAttribute("aria-selected", String(+tr.dataset.id === id)));
    const box = $("detailBody");
    box.className = "";
    box.textContent = "불러오는 중…";
    try {
      const d = await getJSON(`/ai/detections/${id}`);
      const ex = d.explanation || {};
      const st = d.is_anomaly ? "anomaly" : "normal";
      $("detailMeta").textContent = `#${d.id} · ${d.model_version} · ${d.source === "realtime" ? "실시간" : d.source === "backfill" ? "재분석" : "수동"} 분석`;
      const win = ex.window || {};
      const contrib = (ex.top_features || []).map((t) => `<div class="ai-contrib-row"><span>${esc(t.label)}</span><span class="ai-bar"><i style="width:${Math.min(100, t.contribution_pct)}%"></i></span><b>${t.contribution_pct}%</b></div>`).join("");
      const feats = Object.entries(d.features || {}).map(([k, v]) => `<tr><td>${esc(LABELS[k] || k)}</td><td>${esc(fmtFeature(k, v))}</td></tr>`).join("");
      const fbHist = (d.feedback || []).map((f) => `<li><span class="ai-pill" data-v="${esc(f.label)}">${esc({ normal: "정상", anomaly: "이상", hold: "보류" }[f.label])}</span> ${esc(new Date(f.created_at).toLocaleString("ko-KR"))}${f.note ? " · " + esc(f.note) : ""}</li>`).join("");
      box.innerHTML = `<div class="ai-summary" data-status="${st}">${d.is_anomaly ? "🔴 AI 판정: 이상" : "🟢 AI 판정: 정상"} — ${esc(ex.summary || "")}</div>
        <div class="ai-kv"><div>이상 점수<b>${d.score.toFixed(4)}</b></div><div>임계값<b>${d.threshold.toFixed(4)}</b></div><div>점수/임계값<b>×${(d.score / d.threshold).toFixed(2)}</b></div>
        <div>분석 구간 (KST)<b>${esc(kst(win.start))}~${esc(kst(win.end))}</b></div><div>구간 가격 변화<b>${Number.isFinite(win.price_change_pct) ? (win.price_change_pct >= 0 ? "+" : "") + win.price_change_pct.toFixed(3) + "%" : "—"}</b></div><div>종가<b>${won(d.close)}</b></div></div>
        <div class="ai-sub">이상 점수 기여 특징량</div><div class="ai-contrib">${contrib || "—"}</div>
        <div class="ai-sub">분석 근거</div><ul class="ai-evidence">${(ex.evidence || []).map((e) => `<li>${esc(e)}</li>`).join("")}</ul>
        <div class="ai-sub">탐지 시점(마지막 1분) 특징량</div><table class="ai-feat"><tbody>${feats}</tbody></table>
        <div class="ai-sub">사용자 판정</div>
        <div class="ai-fb"><button type="button" data-label="normal">정상</button><button type="button" data-label="anomaly">이상</button><button type="button" data-label="hold">판단 보류</button><input id="fbNote" maxlength="500" placeholder="메모 (선택)"></div>
        <ul class="ai-fb-hist">${fbHist || "<li>아직 판정 기록이 없습니다.</li>"}</ul>
        <p class="ai-disclaimer">${esc(ex.disclaimer || "")}</p>`;
      box.querySelectorAll(".ai-fb button").forEach((b) => b.addEventListener("click", () => sendFeedback(id, b.dataset.label)));
    } catch (err) {
      box.textContent = `상세 정보를 불러오지 못했습니다 (${err.message}).`;
    }
  }
  async function sendFeedback(id, label) {
    const note = ($("fbNote") && $("fbNote").value.trim()) || null;
    try {
      await getJSON(`/ai/detections/${id}/feedback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ label, note }) });
      await Promise.all([loadDetail(id), loadDetections(), refreshStatus()]);
    } catch (err) { alert("판정 저장 실패: " + err.message); }
  }

  // ------------------------------------------------------------- refresh
  async function refreshStatus() {
    const [col, ai, latest, fb] = await Promise.allSettled([getJSON("/status"), getJSON("/ai/status"), getJSON("/ai/analysis/latest"), getJSON("/ai/feedback/summary")]);
    renderCollector(col.status === "fulfilled" ? col.value : null);
    const aiv = ai.status === "fulfilled" ? ai.value : null;
    state.aiStatus = aiv;
    renderModel(aiv);
    renderJudge(latest.status === "fulfilled" ? latest.value : null);
    renderHistory(aiv, fb.status === "fulfilled" ? fb.value : null);
    if (col.status === "rejected" || ai.status === "rejected") { notice("API 서버에 연결할 수 없습니다. 잠시 후 다시 시도하세요.", "error"); return; }
    // Banner only for real outages; normal states (warm-up etc.) live in the status cards.
    const msgs = [];
    if (aiv && !aiv.model_loaded) msgs.push("학습된 AI 모델이 배포되지 않아 이상탐지를 수행하지 않습니다.");
    if (aiv && aiv.model && aiv.model.data_label === "synthetic") msgs.push("현재 모델은 합성(시험용) 데이터로 학습되었습니다.");
    if (aiv && aiv.last_error) console.warn("CheckBit AI:", aiv.last_error);
    notice(msgs.join(" · ") || null);
  }
  async function refreshTimeline() {
    try { state.timeline = await getJSON(`/ai/timeline?minutes=${state.range}`); }
    catch (err) { state.timeline = { points: [], message: `차트 데이터를 불러오지 못했습니다 (${err.message}).` }; }
    drawChart();
  }
  // ------------------------------------------------------------ model card
  const pct = (v, d = 1) => (Number.isFinite(+v) ? (v * 100).toFixed(d) + "%" : "—");
  const num = (v, d = 4) => (Number.isFinite(+v) ? (+v).toFixed(d) : "—");
  async function loadModelInfo() {
    const box = $("modelInfo");
    try {
      const m = await getJSON("/ai/model");
      if (!m.loaded) { box.innerHTML = `<p class="ai-detail-empty">학습된 모델이 배포되지 않았습니다. ${esc(m.error || "")}</p>`; return; }
      const a = m.architecture, d = m.data || {}, t = m.training || {}, ev = m.evaluation || {};
      const w = d.windows || {}, sd = ev.score_distribution || {}, ar = ev.alarm_rate || {};
      const dataLabel = d.label === "synthetic" ? "합성 데이터(테스트용)" : "Upbit WebSocket 실시간 수집";
      const split = (d.split_ratio || []).map((r) => Math.round(r * 100)).join(" / ");
      const dist = ["train", "val", "test"].filter((k) => sd[k] && sd[k].n).map((k) =>
        `<tr><td>${{ train: "학습", val: "검증", test: "테스트" }[k]}</td><td>${sd[k].n}</td><td>${num(sd[k].mean)}</td><td>${num(sd[k].p95)}</td><td>${num(sd[k].max)}</td></tr>`).join("");
      const syn = ((ev.synthetic_injection || {}).results || []).map((r) =>
        `<tr><td>±${r.delta_std}σ</td><td>${pct(r.lstm_detect_rate)}</td><td>${num(r.lstm_auc, 3)}</td><td>${pct(r.baseline_detect_rate)}</td><td>${num(r.baseline_auc, 3)}</td></tr>`).join("");
      box.innerHTML = `<div class="ai-model-grid">
        <div class="ai-model-box"><h4>모델 구조</h4><dl><dt>모델</dt><dd>${esc(a.type)}</dd><dt>인코더 → 디코더</dt><dd>${esc(a.encoder)} → ${esc(a.decoder)}</dd>
          <dt>출력</dt><dd>${esc(a.output)}</dd><dt>파라미터</dt><dd>${Number(a.n_params).toLocaleString("ko-KR")}개</dd><dt>입력 윈도우</dt><dd>${a.window}분 × ${m.features.length}개 특징량</dd><dt>버전</dt><dd>${esc(m.version)}</dd></dl></div>
        <div class="ai-model-box"><h4>학습 데이터</h4><dl><dt>출처</dt><dd>${esc(dataLabel)}</dd><dt>기간 (KST)</dt><dd>${esc(kst(d.first_minute, true))} ~ ${esc(kst(d.last_minute))}</dd>
          <dt>체결 / 1분 봉</dt><dd>${Number(d.trades).toLocaleString("ko-KR")}건 / ${d.minute_bars}개</dd><dt>시간순 분할</dt><dd>${esc(split)}</dd>
          <dt>윈도우 (학습/검증/테스트)</dt><dd>${w.train} / ${w.val} / ${w.test}</dd><dt>최적 epoch</dt><dd>${t.best_epoch ?? "—"} (val MSE ${num(t.best_val_loss)})</dd></dl></div>
        <div class="ai-model-box"><h4>판정 기준</h4><dl><dt>이상 점수</dt><dd>마지막 ${a.score_steps || a.window}분 재구성 오차(MSE)</dd><dt>임계값</dt><dd>${num(m.threshold)}</dd>
          <dt>임계값 산정</dt><dd>${esc(/MAD/.test(m.threshold_method || "") ? "검증 점수 중앙값 + 4·MAD" : (m.threshold_method || "—"))}</dd><dt>정규화</dt><dd>학습 구간 z-score</dd><dt>검증 경보율</dt><dd>${pct(ar.val)}</dd><dt>테스트 경보율</dt><dd>${pct(ar.test)}</dd></dl></div></div>
        <div class="ai-sub">입력 특징량</div><div class="ai-chips">${m.features.map((f) => `<span>${esc(f.label)}</span>`).join("")}</div>
        <div class="ai-eval"><div><div class="ai-sub">재구성 오차 분포</div><table><thead><tr><th>구간</th><th>n</th><th>평균</th><th>p95</th><th>최대</th></tr></thead><tbody>${dist}</tbody></table></div>
          <div><div class="ai-sub">이상 주입 민감도 시험</div><table><thead><tr><th>교란 크기</th><th>LSTM 탐지율</th><th>LSTM AUC</th><th>통계모델 탐지율</th><th>통계모델 AUC</th></tr></thead><tbody>${syn || '<tr><td colspan="5">—</td></tr>'}</tbody></table></div></div>
        <p class="ai-eval-note">테스트 구간 윈도우의 최근 3분에 특징량 3개를 인위적으로 교란해 정상 윈도우와 구분되는지 측정한 값입니다. 실제 이상 라벨이 없어 정확도·정밀도·재현율은 산출하지 않으며, 사용자 판정이 누적되면 AI-사용자 일치율로 검증합니다.</p>`;
    } catch (err) {
      box.innerHTML = `<p class="ai-detail-empty">모델 정보를 불러오지 못했습니다 (${esc(err.message)}).</p>`;
    }
  }
  async function refreshAll() { await Promise.all([refreshStatus(), refreshTimeline(), loadDetections()]); }

  // --------------------------------------------------------------- admin
  const tokenKey = "checkbit_admin_token";
  try { $("adminToken").value = sessionStorage.getItem(tokenKey) || ""; } catch { /* storage blocked */ }
  async function admin(path, okText) {
    const token = $("adminToken").value.trim();
    try { sessionStorage.setItem(tokenKey, token); } catch { /* ignore */ }
    const msg = $("adminMsg");
    if (!token) { msg.textContent = "관리자 토큰을 입력하세요."; return; }
    try {
      const r = await getJSON(path, { method: "POST", headers: { "X-Admin-Token": token } });
      msg.textContent = okText(r);
      await Promise.all([refreshAll(), loadModelInfo()]);
    } catch (err) { msg.textContent = `실패: ${err.message}`; }
  }
  $("colStart").addEventListener("click", () => admin("/collector/start", (r) => r.already_running ? "이미 수집 중입니다." : "수집을 시작했습니다."));
  $("colStop").addEventListener("click", () => admin("/collector/stop", (r) => r.connection_closed ? "수집을 중지하고 WebSocket 연결을 닫았습니다." : "중지 요청됨 (연결 종료 대기 중)"));
  $("modelReload").addEventListener("click", () => admin("/ai/model/reload", (r) => r.loaded ? `모델 ${r.version} 로드 완료 (재분석 ${r.backfilled}건)` : `모델 로드 실패: ${r.error}`));

  // -------------------------------------------------------------- wiring
  $("aiAnalyze").addEventListener("click", async () => {
    const b = $("aiAnalyze"); b.disabled = true;
    try {
      const r = await getJSON("/ai/analyze", { method: "POST" });
      if (r.status !== "ok") renderJudge(r);
      else { await refreshAll(); loadDetail(r.id); }
    } catch (err) { notice(`분석 요청 실패: ${err.message}`, "error"); }
    finally { b.disabled = false; }
  });
  document.querySelectorAll("[data-ai-range]").forEach((btn) => btn.addEventListener("click", () => {
    state.range = +btn.dataset.aiRange;
    document.querySelectorAll("[data-ai-range]").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    refreshTimeline();
  }));
  const sel = $("aiFeature");
  Object.entries(LABELS).forEach(([k, v]) => { const o = document.createElement("option"); o.value = k; o.textContent = v; sel.append(o); });
  sel.addEventListener("change", () => { state.feature = sel.value; drawChart(); });
  const wrap = $("aiChartWrap");
  wrap.addEventListener("pointermove", chartMove);
  wrap.addEventListener("pointerleave", chartLeave);
  wrap.addEventListener("click", chartClick);
  window.addEventListener("resize", () => drawChart());

  loadModelInfo();
  refreshAll().then(() => {
    const m = /^#detection-(\d+)$/.exec(location.hash);
    if (m) loadDetail(+m[1]);
  });
  setInterval(() => { if (!document.hidden) { refreshStatus(); refreshTimeline(); } }, 20000);
  setInterval(() => { if (!document.hidden) loadDetections(); }, 60000);
})();
