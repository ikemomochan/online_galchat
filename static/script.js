let currentDetailIndex = 0;
let gyarumindDetailHistory = [];  // [{...8項目...}, ...]
let scoreHistory = [];            // [total, ...]
let gmChart = null;

document.addEventListener("DOMContentLoaded", () => {
  const chatArea = document.getElementById("chat-area");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message");
  const galImg = document.getElementById("gal-img");

  // タブとビューの切替え
  const tabChart = document.getElementById("tab-chart");
  const tabDetail = document.getElementById("tab-detail");
  const chartView = document.getElementById("chart-view");
  const detailView = document.getElementById("detail-view");

  tabChart.onclick = () => {
    tabChart.classList.add("active");
    tabDetail.classList.remove("active");
    chartView.style.display = "block";
    detailView.style.display = "none";
  };

  tabDetail.onclick = () => {
    tabDetail.classList.add("active");
    tabChart.classList.remove("active");
    chartView.style.display = "none";
    detailView.style.display = "block";
  };

  // 履歴ナビ
  document.getElementById("prev-detail").onclick = () => {
    if (currentDetailIndex > 0) {
      currentDetailIndex--;
      updateDetailView();
    }
  };

  document.getElementById("next-detail").onclick = () => {
    if (currentDetailIndex < gyarumindDetailHistory.length - 1) {
      currentDetailIndex++;
      updateDetailView();
    }
  };

  // ====== 関数定義（ここから） ======
  function addBubble(text, sender = "user") {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${sender}`;
    bubble.innerText = text;
    chatArea.appendChild(bubble);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  // 返答を句点などで分割（配列/単文の両対応）
  function renderGalReply(answer) {
    const emit = (arr) => {
      const seen = new Set();
      arr
        .filter(t => {
          const s = String(t).trim();
          if (!s || seen.has(s)) return false;
          seen.add(s);
          return true;
        })
        .slice(0, 3)
        .forEach((t, i) => {
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
      const parts = answer
        .split(/(?<=[。！？.!?])/)
        .map(s => s.trim())
        .filter(Boolean);
      emit(parts);
    } else {
      addBubble(String(answer ?? ""), "gal");
    }
  }

  function setThinking(thinking = true) {
    galImg.src = thinking ? "/static/gal_thinking.png" : "/static/gal_sample.png";
  }

  function updateAverage(score) {
    const avgElem = document.getElementById("gyarumind-average");
    if (score !== undefined && score !== null && !Number.isNaN(score)) {
      avgElem.textContent = `Ave. GYARU-MIDX：${score}/50💖`;
    } else {
      avgElem.textContent = "";
    }
  }

  function showTrendMessage(msg) {
    const msgEl = document.getElementById("trend-message");
    msgEl.textContent = msg ?? "";
  }

  function updateGyarumind(score) {
    const gmEl = document.getElementById("gm-score");
    gmEl.textContent = (score ?? "--");
  }

  function updateChart(historyArr) {
    const ctx = document.getElementById("gm-chart").getContext("2d");
    if (gmChart) gmChart.destroy();
    gmChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: historyArr.map((_, i) => `#${i + 1}`),
        datasets: [{
          label: "ギャルマイン度📈",
          data: historyArr,
          borderColor: "#e91e63",
          backgroundColor: "#ffeef5",
          tension: 0.3,
          pointRadius: 5,
        }]
      },
      options: {
        scales: { y: { min: 0, max: 50 } },
        responsive: true,
        plugins: { legend: { display: false } }
      }
    });
  }

  function updateDetailView() {
    const indexLabel = document.getElementById("detail-index");
    const table = document.getElementById("gyarumind-detail-table");
    if (!Array.isArray(gyarumindDetailHistory) || gyarumindDetailHistory.length === 0) {
      indexLabel.textContent = "#--";
      table.innerHTML = "<tr><td colspan='2'>まだデータがないよ💦</td></tr>";
      return;
    }
    const detail = gyarumindDetailHistory[currentDetailIndex];
    const excludedKeys = ["レジリエンス", "自他境界"]; // UIから除外
    indexLabel.textContent = `#${(currentDetailIndex + 1)}`;
    table.innerHTML = "";
    for (const [rawKey, value] of Object.entries(detail)) {
      const key = rawKey.trim();
      if (excludedKeys.includes(key)) continue;
      const row = document.createElement("tr");
      row.innerHTML = `<td>${key}</td><td>${value}</td>`;
      table.appendChild(row);
    }
  }
  // ====== 関数定義（ここまで） ======

  // 送信ハンドラ
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    addBubble(text, "user");
    input.value = "";
    input.focus();

    const loadingBubble = document.createElement("div");
    loadingBubble.className = "bubble gal";
    loadingBubble.innerText = "……🤔";
    chatArea.appendChild(loadingBubble);
    chatArea.scrollTop = chatArea.scrollHeight;
    setThinking(true);

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      console.log("レスポンス status:", res.status);
      console.log("レスポンス content-type:", res.headers.get("content-type"));

      const data = await res.json();
      console.log("パースできたJSON:", data);

      loadingBubble.remove();
      renderGalReply(data.answer);

      // === 新API（gmdオブジェクト）にも旧APIにも対応（単一版） ===
      if (data?.gmd) {
        const g = data.gmd;

        // push & UI更新
        scoreHistory.push(g.total);
        gyarumindDetailHistory.push(g.details);
        currentDetailIndex = gyarumindDetailHistory.length - 1;

        updateGyarumind(g.total);
        updateChart(scoreHistory);
        updateDetailView();

        // 平均スコア
        const sum = scoreHistory.reduce((a, b) => a + b, 0);
        const avg = Math.round((sum / scoreHistory.length) * 100) / 100;
        updateAverage(avg);

        // トレンド（前回→今回）
        if (scoreHistory.length >= 2) {
          const last = scoreHistory[scoreHistory.length - 1];
          const prev = scoreHistory[scoreHistory.length - 2];
          const diff = Math.round((last - prev) * 100) / 100;

          // let msg = "横ばい";
          let msg = "Flat";
          const th = 0.25; // ±0.25未満は横ばい扱い
          // if (diff > th) msg = `上昇中（前回比 +${diff.toFixed(2)}）`;
          if (diff > th) msg = `Rising\n（vs Previous +${diff.toFixed(2)}）`;
          // else if (diff < -th) msg = `下降中（前回比 ${diff.toFixed(2)}）`;
          else if (diff < -th) msg = `Falling（vs Previous ${diff.toFixed(2)}）`;

          showTrendMessage(msg);
        } else {
          showTrendMessage(""); // 初回は非表示
        }
      }

      // 旧APIフォールバック
      if (typeof data?.average_score === "number") updateAverage(data.average_score);
      if (typeof data?.trend_message === "string") showTrendMessage(data.trend_message);
    } catch (err) {
      console.error(err);
      loadingBubble.remove();
      addBubble("ごめん、ちょいエラー出たっぽい。もう一回だけ試してみて！", "gal");
      showTrendMessage("通信エラーかも（リトライ推奨）");
    } finally {
      setThinking(false);
    }
  });
}); // ← ここで DOMContentLoaded を “必ず” 閉じる
