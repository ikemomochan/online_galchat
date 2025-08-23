let gyarumindDetailHistory = [];
let currentDetailIndex = 0;
const ADVICE_MAP = {
  "自己受容": "弱さも含めて、あなたはあなたでいいんだよ！",
  "自己肯定感": "人と比べる必要ないよ！そのまんまの君で、もうめっちゃイケてるよ👍",
  "感情の強度": "もっと素直に自分の気持ちさらけ出しちゃっていいよ！誰も見てないし！",
  "言語クリエイティビティ": "もっとおもろい言い方に変えたらテンションアガるかも！",
  "共感・他者リスペクト": "いやなことは“どうしてこうなったんだろう？”って一歩引いて考えてみると、肩の力がふっと抜けるよ",
  "ポジティブ変換力": "今の気持ちはそのままでOK！できたら今日の“ちょいハッピー”考えてみてね！✨",
};

document.getElementById("gyarumind-average")   // 平均スコアの表示先
document.getElementById("trend-message")       // 上昇/下降のメッセージ表示先
document.addEventListener("DOMContentLoaded", () => {
  const chatArea = document.getElementById("chat-area");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message");
  const galImg = document.getElementById("gal-img");

   // ← このへんに追加するとベスト！
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


  function addBubble(text, sender = "user") {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${sender}`;
    bubble.innerText = text;
    chatArea.appendChild(bubble);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  function setThinking(thinking = true) {
    galImg.src = thinking
      ? "/static/gal_thinking.png"
      : "/static/gal_sample.png";
  }

  // 平均スコアを更新
function updateAverage(score) {
   const avgElem = document.getElementById("gyarumind-average");
  if (score !== undefined && score !== null) {
    avgElem.textContent = `平均ギャルマイン度：${score}/50💖`;
  } else {
    avgElem.textContent = "";  // または消す、デフォ表示にするなど
  }
}

// 上下メッセージ
function showTrendMessage(msg) {
  const msgEl = document.getElementById("trend-message");
  msgEl.textContent = msg ?? "";
}


  // ギャルマイン度表示用
  function updateGyarumind(score) {
    const gmEl = document.getElementById("gm-score");
    gmEl.textContent = score ?? "--";
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    addBubble(text, "user");
    input.value = "";
    input.focus();

    const loader = "……🤔";
    const loadingBubble = document.createElement("div");
    loadingBubble.className = "bubble gal";
    loadingBubble.innerText = loader;
    chatArea.appendChild(loadingBubble);
    chatArea.scrollTop = chatArea.scrollHeight;
    setThinking(true);

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      console.log("レスポンス status:", res.status);  // ★ここ追加
      console.log("レスポンス content-type:", res.headers.get("content-type"));  // ★ここ追加


      // if (!res.ok) {
      //   throw new Error(`HTTPエラー: ${res.status}`);
      // }

      const data = await res.json();
      console.log("パースできたJSON:", data);  // ★確認用
      loadingBubble.remove();
      addBubble(data.answer, "gal");

      // ギャルマイン度が返ってきたら表示を更新
      if (data.gyarumind !== undefined && data.gyarumind !== null) {
        updateGyarumind(data.gyarumind);
        updateChart(data.score_history);    // ← グラフ更新
        updateAverage(data.average_score);
        showTrendMessage(data.trend_message); 
      }
      if (data.gyarumind_details_history) {
        gyarumindDetailHistory = data.gyarumind_details_history;
        currentDetailIndex = gyarumindDetailHistory.length - 1; // 最新を表示
        updateDetailView();
      }
      //最低点のものへアドバイス
      if (Array.isArray(data.lowest_items) && data.lowest_items.length) {
  data.lowest_items.forEach(item => {
    // ADVICE_MAP に存在する場合のみ表示
    if (ADVICE_MAP[item]) {
      addBubble(`📢 ${ADVICE_MAP[item]}`, "advice");
    }
  });
}


    } catch (err) {
      console.error("通信エラー詳細:", err); // ← 追加
      loadingBubble.remove();
      addBubble("通信エラーだよ💦", "advice");
    } finally {
      setThinking(false);
    }
  });
});

let gmChart = null;

function updateChart(scoreHistory) {
  console.log("グラフ用データ:", scoreHistory);  // ← ここ！
  const ctx = document.getElementById("gm-chart").getContext("2d");

  // 初回 or 更新のたびにグラフを破棄して描き直す
  if (gmChart) {
    gmChart.destroy();
  }

  gmChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: scoreHistory.map((_, i) => `#${(i + 1) * 5}`),
      datasets: [{
        label: "ギャルマイン度📈",
        data: scoreHistory,
        borderColor: "#e91e63",
        backgroundColor: "#ffeef5",
        tension: 0.3,
        pointRadius: 5,
      }]
    },
    options: {
      scales: {
        y: {
          min: 0,
          max: 50
        }
      },
      responsive: true,
      plugins: {
        legend: { display: false }
      }
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
  const excludedKeys = ["レジリエンス", "自他境界"];

  if (!detail || typeof detail !== "object") return;

  indexLabel.textContent = `#${(currentDetailIndex + 1) * 5}`;

   // マイナス係数の項目（非表示対象）
  //const excludedKeys = ["レジリエンス", "自他境界"];

  table.innerHTML = "";
  for (const [rawKey, value] of Object.entries(detail)) {
  const key = rawKey.trim(); 
  if (excludedKeys.includes(key)) continue;          // ← この行で負寄与項目をスキップ
  const row = document.createElement("tr");
  row.innerHTML = `<td>${key}</td><td>${value}</td>`;
  table.appendChild(row);
  }
}
