document.addEventListener("DOMContentLoaded", () => {
  // 要素の取得
  const chatArea = document.getElementById("chat-area");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message");
  const sendButton = document.querySelector("#chat-form button");
  
  // モーダル関連
  const scoreBtn = document.getElementById("score-btn");
  const modal = document.getElementById("stats-modal");
  const closeModal = document.getElementById("close-modal");
  const headerScore = document.getElementById("header-score");

  // タイマー関連
  const timerDisplay = document.getElementById("timer-display");
  let remainingSeconds = 300; // 初期値（サーバーと同期して補正）

  // --- モーダル操作 ---
  if(scoreBtn) scoreBtn.onclick = () => modal.classList.remove("hidden");
  if(closeModal) closeModal.onclick = () => modal.classList.add("hidden");
  if(modal) modal.onclick = (e) => { if(e.target === modal) modal.classList.add("hidden"); };

  // --- タイマー表示更新 ---
  function updateTimerDisplay() {
      if (!timerDisplay) return;
      
      // デバッグモード（99999秒）の場合は表示を変える
      if (remainingSeconds > 90000) {
          timerDisplay.textContent = "∞ (Dev Mode)";
          return;
      }

      const m = Math.floor(remainingSeconds / 60);
      const s = Math.floor(remainingSeconds % 60);
      timerDisplay.textContent = `残り ${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
      
      if (remainingSeconds <= 0) {
          timerDisplay.textContent = "終了！";
          lockout();
      } else if (remainingSeconds < 30) {
          timerDisplay.style.color = "red";
      }
  }

  // --- 入力禁止（時間切れ時） ---
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

  // --- カウントダウン開始 ---
  const timerInterval = setInterval(() => {
      if (remainingSeconds > 0 && remainingSeconds < 90000) {
          remainingSeconds--;
      }
      updateTimerDisplay();
  }, 1000);

  // --- メッセージ送信処理 ---
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    // 自分の吹き出し追加
    addBubble(text, "user");
    input.value = "";

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      const data = await res.json();

      // 【重要】サーバーからの残り時間で補正
      if (data.remaining_seconds !== undefined) {
          remainingSeconds = Math.floor(data.remaining_seconds);
          updateTimerDisplay();
      }

      // 【重要】強制終了指令が来た場合
      if (data.force_stop) {
          addBubble(data.answer, "gal");
          lockout();
          return;
      }

      // AIの返答を表示
      if (data.answer) {
         // 配列か文字列かで分岐して表示（既存ロジック使用）
         renderGalReply(data.answer, data.intent);
      }

      // スコア更新があればヘッダーに反映
      if (data.gmd) {
          headerScore.textContent = data.gmd.total;
          // グラフ更新関数などを呼ぶ (updateChartなど)
      }

    } catch (err) {
      console.error(err);
      addBubble("通信エラーかも💦", "gal");
    }
  });

  // 吹き出し追加などのヘルパー関数（既存のものを使用）
  function addBubble(text, sender) {
      const div = document.createElement("div");
      div.className = `bubble ${sender}`;
      div.innerHTML = text.replace(/\n/g, "<br>");
      chatArea.appendChild(div);
      chatArea.scrollTop = chatArea.scrollHeight; // 最下部へスクロール
  }
  
  // ... (renderGalReply, updateChart など既存の関数はそのまま維持) ...
});