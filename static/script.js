document.addEventListener("DOMContentLoaded", () => {
  // === 要素の取得 ===
  const chatArea = document.getElementById("chat-area");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message");
  const sendButton = document.querySelector("#chat-form button");
  
  // モーダル関連
  const scoreBtn = document.getElementById("score-btn");
  const modal = document.getElementById("stats-modal");
  const closeModal = document.getElementById("close-modal");
  const headerScore = document.getElementById("header-score");
  const headerGalImg = document.getElementById("header-gal-img");
  
  // グラフ用キャンバス
  const ctx = document.getElementById("gm-chart") ? document.getElementById("gm-chart").getContext("2d") : null;
  let gmChart = null;

  // タイマー関連
  const timerDisplay = document.getElementById("timer-display");
  let remainingSeconds = null; 
  let isDevMode = false;

  // 履歴データ
  let scoreHistory = [];
  let detailHistory = [];
  let currentDetailIndex = 0;
  const detailLabelMap = {
    "自己肯定感": "Self-Esteem",
    "自己受容": "Self-Acceptance",
    "楽観性": "Optimism",
    "自他境界": "Self-Other Boundary",
    "本来性": "Authenticity",
    "他者尊重": "Other-Respect",
    "感情の強度": "Emotional Intensity",
    "言語創造性": "Linguistic Creativity",
  };

  // Metric tabs / per-metric history
  const metricTabsContainer = document.getElementById("metric-tabs");
  const metricDetailTable = document.getElementById("metric-detail-table");
  const metricAverageEl = document.getElementById("metric-average");
  let activeMetric = null;
  let metricHistory = {}; // { metricKey: [values...] }

  function formatNumber(v, digits=2) {
    if (v === null || v === undefined) return "-";
    return Number(v).toFixed(digits);
  }

  function computeAverage(arr) {
    if (!Array.isArray(arr) || arr.length === 0) return null;
    const sum = arr.reduce((a,b)=>a+Number(b||0),0);
    return sum / arr.length;
  }

  function rebuildMetricHistory() {
    metricHistory = {};
    for (const key of Object.keys(detailLabelMap)) {
      metricHistory[key] = [];
    }
    for (const detail of detailHistory) {
      for (const key of Object.keys(detailLabelMap)) {
        metricHistory[key].push(detail[key] !== undefined ? detail[key] : null);
      }
    }
  }

  function buildMetricTabs() {
    if (!metricTabsContainer) return;
    metricTabsContainer.innerHTML = "";
    // add Total tab first
    const totalBtn = document.createElement('button');
    totalBtn.className = 'metric-tab';
    totalBtn.dataset.key = 'TOTAL';
    totalBtn.textContent = 'Total';
    totalBtn.onclick = () => {
      activeMetric = 'TOTAL';
      Array.from(metricTabsContainer.children).forEach(c => c.classList.toggle('active', c === totalBtn));
      renderMetricView('TOTAL');
    };
    metricTabsContainer.appendChild(totalBtn);
    Object.entries(detailLabelMap).forEach(([jp, en]) => {
      const btn = document.createElement('button');
      btn.className = 'metric-tab';
      btn.dataset.key = jp;
      btn.textContent = en;
      btn.onclick = () => {
        activeMetric = jp;
        // highlight
        Array.from(metricTabsContainer.children).forEach(c => c.classList.toggle('active', c === btn));
        renderMetricView(jp);
      };
      metricTabsContainer.appendChild(btn);
    });
    // select first by default
    if (!activeMetric) {
      activeMetric = 'TOTAL';
      if (metricTabsContainer.firstChild) metricTabsContainer.firstChild.classList.add('active');
    }
  }

  function renderMetricView(metricKey) {
    if (!metricDetailTable) return;
    metricDetailTable.innerHTML = '';
    const header = document.createElement('tr');
    if (metricKey === 'TOTAL') {
      header.innerHTML = `<th>Eval</th><th>GYARU-MIDX Total</th>`;
      metricDetailTable.appendChild(header);
      for (let i = 0; i < scoreHistory.length; i++) {
        const row = document.createElement('tr');
        const evalIdx = i + 1;
        const total = scoreHistory[i] !== undefined ? scoreHistory[i] : null;
        row.innerHTML = `<td>#${evalIdx}</td><td>${total !== null ? total : '-'}</td>`;
        metricDetailTable.appendChild(row);
      }
    } else {
      header.innerHTML = `<th>Eval</th><th>${detailLabelMap[metricKey] || metricKey}</th><th>GYARU-MIDX Total</th>`;
      metricDetailTable.appendChild(header);
      for (let i = 0; i < Math.max(metricHistory[metricKey]?.length||0, scoreHistory.length); i++) {
        const row = document.createElement('tr');
        const evalIdx = i + 1;
        const mVal = (metricHistory[metricKey] && metricHistory[metricKey][i] !== undefined) ? metricHistory[metricKey][i] : null;
        const total = scoreHistory[i] !== undefined ? scoreHistory[i] : null;
        row.innerHTML = `<td>#${evalIdx}</td><td>${mVal !== null ? mVal : '-'}</td><td>${total !== null ? total : '-'}</td>`;
        metricDetailTable.appendChild(row);
      }
    }

    // average and trend (English)
    const avg = computeAverage(scoreHistory);
    let prevAvg = null;
    if (scoreHistory.length > 1) prevAvg = computeAverage(scoreHistory.slice(0, -1));
    let avgText = avg !== null ? `Average GYARU-MIDX: ${formatNumber(avg,2)}` : `Average GYARU-MIDX: -`;
    let trendText = '';
    if (prevAvg === null) {
      trendText = 'Trend: N/A (not enough data)';
    } else {
      if (avg > prevAvg) trendText = `Trend: Up ↑ (previous avg: ${formatNumber(prevAvg,2)})`;
      else if (avg < prevAvg) trendText = `Trend: Down ↓ (previous avg: ${formatNumber(prevAvg,2)})`;
      else trendText = `Trend: No change (previous avg: ${formatNumber(prevAvg,2)})`;
    }
    if (metricAverageEl) metricAverageEl.textContent = `${avgText} — ${trendText}`;
  }

  // Panel switching (Breakdown / Trend)
  function switchPanel(name) {
    const tabBreak = document.getElementById('tab-breakdown');
    const tabTrend = document.getElementById('tab-trend');
    const panelBreak = document.getElementById('panel-breakdown');
    const panelTrend = document.getElementById('panel-trend');
    if (!panelBreak || !panelTrend) return;
    if (name === 'trend') {
      panelBreak.style.display = 'none';
      panelTrend.style.display = '';
      if (tabBreak) tabBreak.classList.remove('active');
      if (tabTrend) tabTrend.classList.add('active');
      // ensure metric UI is built when switching to trend
      rebuildMetricHistory();
      buildMetricTabs();
      renderMetricView(activeMetric);
    } else {
      panelBreak.style.display = '';
      panelTrend.style.display = 'none';
      if (tabTrend) tabTrend.classList.remove('active');
      if (tabBreak) tabBreak.classList.add('active');
    }
  }


  // === イベントリスナー設定 ===

  // モーダル操作
  if(scoreBtn) scoreBtn.onclick = () => { modal.classList.remove("hidden"); switchPanel('breakdown'); };
  if(closeModal) closeModal.onclick = () => modal.classList.add("hidden");
  if(modal) modal.onclick = (e) => { if(e.target === modal) modal.classList.add("hidden"); };

  // panel tab buttons
  const tabBreakBtn = document.getElementById('tab-breakdown');
  const tabTrendBtn = document.getElementById('tab-trend');
  if (tabBreakBtn) tabBreakBtn.onclick = () => switchPanel('breakdown');
  if (tabTrendBtn) tabTrendBtn.onclick = () => switchPanel('trend');

  // 詳細切り替えボタン
  const prevBtn = document.getElementById("prev-detail");
  const nextBtn = document.getElementById("next-detail");
  if(prevBtn) prevBtn.onclick = () => { if(currentDetailIndex > 0) { currentDetailIndex--; updateDetailView(); }};
  if(nextBtn) nextBtn.onclick = () => { if(currentDetailIndex < detailHistory.length - 1) { currentDetailIndex++; updateDetailView(); }};


  // === 関数定義 ===

  // 吹き出し追加
  function addBubble(text, sender) {
      const div = document.createElement("div");
      div.className = `bubble ${sender}`;
      div.innerHTML = text.replace(/\n/g, "<br>");
      chatArea.appendChild(div);
      chatArea.scrollTop = chatArea.scrollHeight; 
  }

  // ギャルの返信表示
  function renderGalReply(answer, intent) {
    if (headerGalImg && intent) {
        // 必要ならここで画像切り替え
    }

    const emit = (arr) => {
      arr.forEach((t, i) => {
          if (i === 0) {
            addBubble(t, "gal");
          } else {
            setTimeout(() => addBubble(t, "gal"), i * 1000);
          }
        });
    };

    if (Array.isArray(answer)) {
      emit(answer);
    } else if (typeof answer === "string") {
      const parts = answer.split(/(?<=[。！？.!?])\s*/).filter(Boolean);
      emit(parts);
    } else {
      addBubble(String(answer ?? ""), "gal");
    }
  }

  // タイマー表示更新
  function updateTimerDisplay() {
      if (!timerDisplay) return;
      if (remainingSeconds === null) {
          timerDisplay.textContent = "Unlimited";
          timerDisplay.style.color = "";
          isDevMode = false;
          return;
      }

      
      if (remainingSeconds > 90000) {
          timerDisplay.textContent = "∞ (Dev Mode)";
          isDevMode = true;
          return;
      }

      const m = Math.floor(remainingSeconds / 60);
      const s = Math.floor(remainingSeconds % 60);
      timerDisplay.textContent = `残り ${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
      
      if (remainingSeconds <= 0 && !isDevMode) {
          timerDisplay.textContent = "終了！";
          lockout();
      } else if (remainingSeconds < 30) {
          timerDisplay.style.color = "red";
      }
  }

  // 入力禁止
  function lockout() {
      if(input) {
          input.disabled = true;
          input.placeholder = "体験時間は終了しました🙏";
      }
      if(sendButton) {
          sendButton.disabled = true;
          sendButton.style.background = "#ccc";
      }
  }

  // グラフ更新
  function updateChart(historyArr) {
    if (!ctx) return;
    if (gmChart) gmChart.destroy();
    
    gmChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: historyArr.map((_, i) => `#${i + 1}`),
        datasets: [{
          label: "ギャルマインド度",
          data: historyArr,
          borderColor: "#e91e63",
          backgroundColor: "#ffeef5",
          tension: 0.3,
          pointRadius: 4,
        }]
      },
      options: {
        scales: { y: { min: 0, max: 50 } },
        responsive: true,
        plugins: { legend: { display: false } }
      }
    });
  }

  // 詳細テーブル更新
  function updateDetailView() {
    const indexLabel = document.getElementById("detail-index");
    const table = document.getElementById("gyarumind-detail-table");
    if (!table) return;

    if (!Array.isArray(detailHistory) || detailHistory.length === 0) {
      if(indexLabel) indexLabel.textContent = "#--";
      table.innerHTML = "<tr><td colspan='2'>データなし</td></tr>";
      return;
    }

    const detail = detailHistory[currentDetailIndex];
    if(indexLabel) indexLabel.textContent = `#${(currentDetailIndex + 1)}`;
    
    table.innerHTML = "";
    for (const [key, value] of Object.entries(detail)) {
        if (key === "total") continue; 
        const row = document.createElement("tr");
        const label = detailLabelMap[key] || key;
        row.innerHTML = `<td>${label}</td><td>${value}</td>`;
        table.appendChild(row);
    }
  }


  // === タイマー開始 ===
  const timerInterval = setInterval(() => {
      if (remainingSeconds !== null && remainingSeconds > 0 && remainingSeconds < 90000) {
          remainingSeconds--;
      }
      updateTimerDisplay();
  }, 1000);


  // === 送信処理 ===
  form.addEventListener("submit", async (e) => {
    e.preventDefault(); // ★ここでリロードを防いでいます！
    const text = input.value.trim();
    if (!text) return;

    addBubble(text, "user");
    input.value = "";

    const loadingDiv = document.createElement("div");
    loadingDiv.className = "bubble gal";
    loadingDiv.innerText = "……🤔";
    chatArea.appendChild(loadingDiv);
    chatArea.scrollTop = chatArea.scrollHeight;

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      const data = await res.json();
      loadingDiv.remove();

      if (data.remaining_seconds !== undefined) {
          remainingSeconds = data.remaining_seconds === null ? null : Math.floor(data.remaining_seconds);
          updateTimerDisplay();
      }

      if (data.force_stop) {
          renderGalReply(data.answer, "gal");
          lockout();
          return;
      }

      if (data.answer) {
         renderGalReply(data.answer, data.intent);
      }

      if (data.gmd) {
          if(headerScore) headerScore.textContent = data.gmd.total;
          scoreHistory.push(data.gmd.total);
          detailHistory.push(data.gmd.details);
          currentDetailIndex = detailHistory.length - 1;
          updateChart(scoreHistory);
          updateDetailView();

          // rebuild metric history and tabs, then render current metric view
          rebuildMetricHistory();
          buildMetricTabs();
          renderMetricView(activeMetric);

          // update a short overall trend message (English)
          const trendEl = document.getElementById('trend-message');
          if (trendEl) {
            const last = scoreHistory[scoreHistory.length - 1];
            const prev = scoreHistory.length > 1 ? scoreHistory[scoreHistory.length - 2] : null;
            let msg = `Latest GYARU-MIDX: ${formatNumber(last,2)}`;
            if (prev === null) msg += ' — Trend: N/A';
            else if (last > prev) msg += ` — Trend: Up ↑ (previous: ${formatNumber(prev,2)})`;
            else if (last < prev) msg += ` — Trend: Down ↓ (previous: ${formatNumber(prev,2)})`;
            else msg += ' — Trend: No change';
            trendEl.textContent = msg;
          }
      }

    } catch (err) {
      console.error(err);
      loadingDiv.remove();
      addBubble("通信エラーかも💦（リロードしてみて！）", "gal");
    }
  });

});
