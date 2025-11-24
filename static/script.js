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
  const galImg = document.getElementById("gal-img"); // 旧レイアウト互換用
  
  // グラフ用キャンバス
  const ctx = document.getElementById("gm-chart") ? document.getElementById("gm-chart").getContext("2d") : null;
  let gmChart = null;

  // タイマー関連
  const timerDisplay = document.getElementById("timer-display");
  let remainingSeconds = 300; // 初期値（サーバーと同期して補正）
  let isDevMode = false;

  // 履歴データ
  let scoreHistory = [];
  let detailHistory = [];
  let currentDetailIndex = 0;


  // === イベントリスナー設定 ===

  // モーダル操作
  if(scoreBtn) scoreBtn.onclick = () => modal.classList.remove("hidden");
  if(closeModal) closeModal.onclick = () => modal.classList.add("hidden");
  if(modal) modal.onclick = (e) => { if(e.target === modal) modal.classList.add("hidden"); };

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
      chatArea.scrollTop = chatArea.scrollHeight; // 最下部へスクロール
  }

  // ギャルの返信表示（配列・文字列両対応）
  function renderGalReply(answer, intent) {
    // intentに応じてアイコン切り替え（あれば）
    if (headerGalImg) {
        // アイコンを変える画像があればここで切り替え可能
        // headerGalImg.src = `/static/gyaru_${intent}.png`; 
    }

    const emit = (arr) => {
      const seen = new Set();
      arr
        .filter(t => {
          const s = String(t).trim();
          if (!s || seen.has(s)) return false;
          seen.add(s);
          return true;
        })
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
      // 句点などで分割して表示
      const parts = answer.split(/(?<=[。！？.!?])\s*/).filter(Boolean);
      emit(parts);
    } else {
      addBubble(String(answer ?? ""), "gal");
    }
  }

  // タイマー表示更新
  function updateTimerDisplay() {
      if (!timerDisplay) return;
      
      // デバッグモード（99999秒以上）
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

  // 入力禁止（時間切れ時）
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
        if (key === "total") continue; // 合計点は除外
        const row = document.createElement("tr");
        row.innerHTML = `<td>${key}</td><td>${value}</td>`;
        table.appendChild(row);
    }
  }


  // === タイマー開始 ===
  const timerInterval = setInterval(() => {
      if (remainingSeconds > 0 && remainingSeconds < 90000) {
          remainingSeconds--;
      }
      updateTimerDisplay();
  }, 1000);


  // === 送信処理 ===
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    // 自分の吹き出し追加
    addBubble(text, "user");
    input.value = "";

    // 読み込み中バブル
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

      // 残り時間補正
      if (data.remaining_seconds !== undefined) {
          remainingSeconds = Math.floor(data.remaining_seconds);
          updateTimerDisplay();
      }

      // 強制停止指令
      if (data.force_stop) {
          renderGalReply(data.answer, "gal"); // 最後の挨拶
          lockout();
          return;
      }

      // 通常返答
      if (data.answer) {
         renderGalReply(data.answer, data.intent);
      }

      // スコア更新
      if (data.gmd) {
          // ヘッダー更新
          if(headerScore) headerScore.textContent = data.gmd.total;
          
          // 履歴追加
          scoreHistory.push(data.gmd.total);
          detailHistory.push(data.gmd.details);
          currentDetailIndex = detailHistory.length - 1;
          
          // グラフと詳細テーブル更新
          updateChart(scoreHistory);
          updateDetailView();
          
          // トレンドメッセージ（もしあれば）
          const trendMsg = document.getElementById("trend-message");
          if(trendMsg && scoreHistory.length >= 2) {
             const diff = scoreHistory[scoreHistory.length-1] - scoreHistory[scoreHistory.length-2];
             if(diff > 0.5) trendMsg.textContent = `前回よりアゲ⤴ (+${diff.toFixed(1)})`;
             else if(diff < -0.5) trendMsg.textContent = `ちょいサゲ⤵ (${diff.toFixed(1)})`;
             else trendMsg.textContent = "キープ中✨";
          }
      }

    } catch (err) {
      console.error(err);
      loadingDiv.remove();
      addBubble("通信エラーかも💦（リロードしてみて！）", "gal");
    }
  });

});