import re
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from openai import OpenAI  
from dotenv import load_dotenv                   

# ──────────────────────────
load_dotenv()  # .env を読み込む

app = Flask(__name__, static_folder="static", template_folder="templates")
client = OpenAI(                              # ★client を生成
    api_key=os.getenv("OPENAI_API_KEY"),
)

gyarumind_scores: dict[str, list[float]] = {}
gyarumind_details_history: dict[str, list[dict]] = {}  # ← これ追加！
NEGATIVE_KEYS = {"レジリエンス", "自他境界"}  # ← 1行追加


# プロンプト
SYSTEM_PROMPT = """あなたはポジティブでフレンドリーな女子大生ギャル AI 💖
私の親友になりきって、おしゃべりに付き合ってね！
#お返事の条件
・長文にせず、に簡潔にすること

# 性格
・自己肯定感高め、自分で自分を褒める
・ポジティブな側面を見出すのが得意
・感情的
・スラングや独特なオリジナルギャル語を使う（でも文脈はちゃんとわきまえる）
・人にリスペクトを持っている
・違うと思ったことはハッキリ言える
・相手の話はちゃんと聞く

#しゃべり方
・形容詞・擬音語や感情に君をつける謎の言い回しを多用（「ワクワクくん」「かなしみくん」「うれしみくん」「ぴかぴかくん」「すごすぎくん」など）
・意味不明な誇張表現（「それ聞いた瞬間、ハシビロコウすら踊りだしかねないテンションで」「マジでやばすぎて100億回死に過ぎた猫って話なんだけど」「ヒカ的ノーベル平和賞受」「5億回○○」など）も多く、3文以内でテンポよく話す。
・他にも独自の言葉をよく勝手につくる
"""

GMD_PROMPT = """
以下はユーザーの発言ログです。ギャルマインドを構成する以下の8項目について、それぞれ0〜5点で評価してください。
出力は「項目名: 点数」という形式で、合計点やその他の説明文は不要です。以下は項目名＋（項目の説明）ですが、出力時は項目の説明はカットすること
評価項目:
1. 自己肯定感
2. 自己受容 
3. ポジティブ変換力 
4. レジリエンス (気持ちの切り替え力)
5. 自他境界 （人は人、自分は自分）
6. 共感・他者リスペクト 
7. 感情の強度 
8. 言語クリエイティビティ


発言ログ:
{user_texts}
"""

# ─── メモリにだけ保存（簡易版・ユーザー別） ───
histories: dict[str, list[dict]] = {}

def get_history(sid: str, limit: int = 20) -> list[dict]:
    """ユーザー別履歴を返す（ない場合は空リスト登録）"""
    return histories.setdefault(sid, [])[-limit:]

# ─── ヘルパー関数 ───
import re

def parse_scores(text: str) -> dict[str, float]:
    """
    GPTの出力から8項目スコアを抽出。
    「1. 自己受容: 4」または「自己受容: 4」どちらにも対応します。
    """
    scores: dict[str, float] = {}
    for line in text.splitlines():
        # 先頭の「番号. 」があってもなくてもキャッチする
        m = re.match(r"\s*(?:\d+\.\s*)?(.+?)[:：]\s*([0-5](?:\.\d+)?)", line)
        if not m:
            continue
        key = m.group(1).strip()         # 「自己受容」など
        val = float(m.group(2))          # 数字部分
        scores[key] = val
    return scores


def estimate_gyarumind(user_texts: list[str]) -> float | None:
    """最新5発言でギャルマイン度を重回帰式で計算"""
    if len(user_texts) < 5:
        return None

    prompt = GMD_PROMPT.format(user_texts="\n".join(user_texts[-5:]))
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        content = res.choices[0].message.content
        scores = parse_scores(content)

        weights = {
            "自己受容": 4.888,
            "自己肯定感": 4.681,
            "感情の強度": 3.837,
            "言語クリエイティビティ": 3.403,
            "共感・他者リスペクト": 2.601,
            "ポジティブ変換力": 0.916,
            "レジリエンス": -1.914,
            "自他境界": -3.541,
        }
        intercept = -20.33430342
        total = sum(scores.get(k, 0) * w for k, w in weights.items()) + intercept
        return {
            "total": round(total, 2),
            "details": scores  # ← これが8項目
        }
    except Exception as e:
        print(f"[{datetime.now()}] ギャルマイン度推定エラー: {e}")
        return None

# ─── ルート定義 ───
@app.route("/")
def index():
    return render_template("gal_index.html")

@app.route("/ask", methods=["POST"])
def ask():
    sid = request.cookies.get("sid") or request.remote_addr
    history = histories.setdefault(sid, [])
    gyarumind_scores.setdefault(sid, [])
    # ← ここで初期化！
    lowest_items: list[str] = []
    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"answer": "え？なんて？💦"})

    history.append({"role": "user", "content": user_msg})

    # ChatCompletion
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *get_history(sid)]
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.8,
            max_tokens=180,
        )
        answer = res.choices[0].message.content.strip()
    except Exception as e:
        answer = f"エラー起きちゃった💦 ({e})"

    history.append({"role": "assistant", "content": answer})

    # ─── ここからギャルマイン度計算 ───
    user_texts = [m["content"] for m in history if m["role"] == "user"]
    gyarumind = None
    trend_message = None

    if len(user_texts) >= 5 and len(user_texts) % 5 == 0:
        result = estimate_gyarumind(user_texts)
        if result is not None:
            gyarumind_scores.setdefault(sid, []).append(result["total"])
            gyarumind_details_history.setdefault(sid, []).append(result["details"])
            gyarumind = result["total"]

            # 最低スコアを探す
            details = result["details"]
            min_score = min(details.values())
            lowest_items = [
            k for k, v in details.items()
            if v == min_score and k not in NEGATIVE_KEYS     # ★ ここで除外
        ]

    score_list = gyarumind_scores[sid]
    average_score = round(sum(score_list) / len(score_list), 2) if score_list else None

    if len(score_list) >= 2:
        if score_list[-1] > score_list[-2]:
            trend_message = "ギャルマイン度上昇中↑ いい感じ〜💖"
        elif score_list[-1] < score_list[-2]:
            trend_message = "ギャルマイン度下降中↓ 無理しないでね💦"
        else:
            trend_message = "ギャルマイン度変わらず〜😌"

    return jsonify({
        "answer": answer,
        "gyarumind": gyarumind,
        "score_history": score_list,
        "average_score": average_score,
        "trend_message": trend_message,
        "gyarumind_details_history": gyarumind_details_history.get(sid, []),
        "lowest_items": lowest_items
    }), 200



# ──────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
