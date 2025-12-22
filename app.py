import os, json, time, glob, random, hashlib, re
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = "gal_experiment_final_ver"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 設定 ===
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_NAME = "gpt-4.1-mini" # コスト・速度バランス重視

# === ②-d PLS回帰用係数 ===
CONST = {
    "std_beta": [1.106135, 1.370707, 0.874889, -0.04322, 0.477184, 0.133321, 1.535123, 1.133268],
    "mu_y": 26.61, "sigma_y": 3.89,
    "mu_xs": [3.39, 3.51, 3.45, 3.72, 3.77, 3.90, 2.87, 1.06],
    "sigma_xs": [0.62, 0.58, 0.65, 0.59, 0.69, 0.67, 0.89, 0.80],
}
FEATURE_KEYS = ["自己肯定感", "自己受容", "楽観性", "自他境界", "本来性", "他者尊重", "感情の強度", "言語創造性"]
FEATURE_DESCRIPTIONS = {
            "自己肯定感": "自分をどれだけ「良い/価値がある」と評価しているか",
            "自己受容": "長所も短所も含め、あるがままの自分を受け入れる姿勢",
            "楽観性": "物事がうまく進むと一般的に期待する傾向",
            "自他境界": "感情に巻き込まれず・同一化/遮断に偏らず立場を保てる傾向",
            "本来性": "外圧に過度に左右されず価値観に沿って選ぶ傾向",
            "他者尊重": "相手の価値・個性・尊厳を尊重する態度",
            "感情の強度": "強調語/感嘆の多さなど感情表出の強さ",
            "言語創造性": "スラング/造語/比喩等の創造的表現",
        }

# === ②-c 採点プロンプト用ルーブリック ===
METRIC_RUBRICS = {
    "自己肯定感": {
        "scale": {
            0: "強い自己否定/無価値感",
            1: "自分に対して低評価が多い",
            2: "部分的自己否定",
            3: "中立",
            4: "自分の価値をある程度認める",
            5: "積極的な自己肯定。"
        },
        "pos_examples": ["「私は自分に良い資質がいくつもあると感じている」", "「自分に満足している」", "「私はたいていの人と同じくらい上手に物事をこなせる。」"],
        "neg_examples": ["「ときどき、私はまったくダメだと思うことがある。」", "「誇れることがあまりないと感じる。」", "「自分が役に立たないと感じる」"]
    },
    "自己受容": {
        "scale": {
            0: "自己に対する強い拒否",
            1: "自分を受容できない発言が多い",
            2: "部分的に非受容",
            3: "中立",
            4: "自分について概ね受容",
            5: "ほぼ全面的に自分を受容。"
        },
        "pos_examples": ["「自分の欠点を受け入れられる」", "「他人の期待に応えられなくても、自分を価値ある存在として見なせる」"],
        "neg_examples": ["「失敗すると、自分には価値がないように感じる。」", "「他人から拒絶されると、自分が嫌いになる。」"]
    },
    "楽観性": {
        "scale": {
            0: "極めて悲観",
            1: "悲観",
            2: "やや悲観",
            3: "中立",
            4: "やや楽観",
            5: "強い楽観"
        },
        "pos_examples": ["「不確実な状況でも、たいてい最善の結果を期待する。」", "「私は自分の将来について常に楽観的である。」"],
        "neg_examples": ["「物事が自分の思い通りにいくとはほとんど期待しない。」", "「良いことが自分に起こるとはめったに期待しない。」"]
    },
    "自他境界": {
        "scale": {
            0: "自他境界が著しく曖昧/過剰遮断",
            1: "自他の同一化/遮断が頻繁",
            2: "やや不安定",
            3: "中立",
            4: "概ね健全",
            5: "明確で柔軟に一貫。"
        },
        "pos_examples": ["「感情的な相手にも一呼吸」", "「周囲の期待を尊重しつつ自分の考えを伝える」", "「衝突しても関係を切らずに、話し合いを続けようとする」"],
        "neg_examples": ["「相手の機嫌で判断が揺れる」", "「すぐ意見変更/連絡遮断」", "「課題を自分と混同」"]
    },
    "本来性": {
        "scale": {
            0: "強い迎合・自己疎外（他者の目に全面依存）",
            1: "迎合的で、自分らしさを優先しにくい",
            2: "時に流される",
            3: "中立",
            4: "概ね価値観に沿う",
            5: "一貫して沿う。"
        },
        "pos_examples": ["「多数派と違っても、自分の価値観に沿って選ぶ理由を落ち着いて説明できる。」", "「評価やトレンドよりも“自分が大切と思う軸”に基づいて継続的に選択している。」"],
        "neg_examples": ["「“人にどう見られるか”で選択が大きく変わる/すぐ迎合してしまう。」", "「相手の機嫌や評価に合わせて、自分の本心と逆の選択をすることが多い。」"]
    },
    "他者尊重": {
        "scale": {
            0: "他者を常に見下し/軽視",
            1: "他者への尊重弱く偏見多い",
            2: "他者を時に尊重できない",
            3: "中立",
            4: "他者を概ね尊重",
            5: "他者を一貫して尊重。"
        },
        "pos_examples": ["「違いがあっても価値を認める」", "「相手の個性や選択を理解しようとし、丁寧に意見を扱う」"],
        "neg_examples": ["「価値がないと烙印」", "「権利無視して押し通す」"]
    },
    "感情の強度": {
        "scale": {
            0: "強調なし",
            1: "「とても」など、穏やかな強調のみ",
            2: "多少の強調",
            3: "強調+「！」",
            4: "強調多用/感嘆",
            5: "極端な誇張/多用。"
        },
        "pos_examples": ["「ヤバすぎ」", "「〜すぎる！」", "「〇〇すぎて死ぬ」", "「エグイ」"],
        "neg_examples": ["事務的・儀礼的な表現のみ。"]
    },
    "言語創造性": {
        "scale": {
            0: "遊び皆無",
            1: "わずかに砕けた表現",
            2: "既存スラング・砕けた表現がやや見受けられる",
            3: "スラングや砕けた表現が自然に見受けられる",
            4: "発言が終始くだけたイメージで、スラングや造語も多数",
            5: "全ての発言で造語やスラングを多く使ってる"
        },
        "pos_examples": ["「今日も『ピカピカくん』な1日にしよーね！」", "「まって、それって超絶キラキラピーナッツバターすぎてむり」"],
        "neg_examples": ["「とても光栄に思います。」", "「嬉しくないと言ったらなりますね」"]
    }
}

# === ①-b Few-Shot データ ===
FEW_SHOTS = {
  "sympathy": [
    {"role":"user","content":"顔とか性格タイプの人と付き合うのむずくないですか！？"},
    {"role":"assistant","content":"むずすぎな、普通に。誰か教えてーて感じ。"},
    {"role":"user","content":"幸せって何なのか分からなくなった"},
    {"role":"assistant","content":"周りがみんな生きてること。"},
  ],
  "advice": [
    {"role":"user","content":"同じ人に告白、何回断られたら諦めるべき？"},
    {"role":"assistant","content":"回数を決めるより、燃えるだけ燃えてこ～🔥🔥ってかんじ"},
    {"role":"user","content":"好きな人を夏祭りに誘いたいんですけど、どうしたらいいと思いますか！"},
    {"role":"assistant","content":"可愛い！素敵すぎる！行きたいって誘ったら？普通に。"},
  ],
  "energy": [
    {"role":"user","content":"疲れてテンション上がらん"},
    {"role":"assistant","content":"スタバのグランデ飲みましょ"},
    {"role":"user","content":"おはよー"},
    {"role":"assistant","content":"おはよー❕❕❕🌞🌞"},
  ]
}

# アドバイス用テキスト（最低点項目用）
ADVICE_TEXTS = {
    "自己肯定感": "もっと自分を褒めてあげて！昨日の自分より1mmでも進んでればOKだし💖",
    "自己受容": "ダメな自分も「ウケるw」って笑い飛ばしちゃお！完璧じゃなくていいんだよ。",
    "楽観性": "「なんとかなるっしょ！」を口癖にしてみて！意外とマジでなんとかなるから✨",
    "自他境界": "人の機嫌は人のもの！自分の機嫌をとることだけに集中しよ！",
    "本来性": "「本当はどうしたい？」って自分に聞いてみて。直感信じてこ！",
    "他者尊重": "相手の話を「へー！」って聞くだけでOK。同意しなくても尊重はできるよ！",
    "感情の強度": "嬉しいときは全力で喜んで、悲しいときは思いっきり泣こ！感情は出し切るのが吉👍",
    "言語創造性": "もっとヤバい言葉使ってこ！自分の気持ちにピッタリな言葉、作っちゃえ！"
}

# === ユーティリティ関数 ===

def _std_vec(x_vals: List[float]) -> np.ndarray:
    mu = np.array(CONST["mu_xs"], float)
    sd = np.array(CONST["sigma_xs"], float)
    x  = np.array(x_vals, float)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd

def _pls_y_from_details(details_model_keys: Dict[str, float]) -> float:
    # 辞書からリストへ変換（FEATURE_KEYS順）
    x = [float(details_model_keys.get(k, 0.0)) for k in FEATURE_KEYS]
    z = _std_vec(x)
    beta = np.array(CONST["std_beta"], float)
    z_pred = float(z @ beta)
    y = float(CONST["mu_y"] + CONST["sigma_y"] * z_pred)
    return max(0.0, min(50.0, y))

def get_user_file(uid): return os.path.join(DATA_DIR, f"{uid}.json")

def load_user_data(uid):
    filepath = get_user_file(uid)
    if not os.path.exists(filepath): return None 
    with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)

def save_user_data(uid, data):
    with open(get_user_file(uid), 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)

def check_lockout(last_finished_at):
    if last_finished_at == 0: return False, 0
    elapsed = time.time() - last_finished_at
    lockout_time = 18 * 60 * 60 
    if elapsed < lockout_time: return True, lockout_time - elapsed
    return False, 0

# === ① レスポンス生成ロジック ===

def classify_intent(user_msg: str) -> str:
    """①-a: ユーザーの返答から文脈判断"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "出力は advice, sympathy, energy のいずれか1単語のみ。"},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.0, max_tokens=10
        )
        label = res.choices[0].message.content.strip().lower()
        return label if label in FEW_SHOTS else "energy"
    except: return "energy"

def sample_shots(intent: str, k: int = 2) -> List[Dict]:
    """①-b: 文脈に合ったFew-Shotを取得"""
    pool = FEW_SHOTS.get(intent, FEW_SHOTS["energy"])
    pairs = [(pool[i], pool[i+1]) for i in range(0, len(pool), 2) if i+1 < len(pool)]
    if not pairs: return []
    random.shuffle(pairs)
    picked = pairs[:k]
    return [m for p in picked for m in p]

def generate_reply(user_msg, data):
    """①-b: システムプロンプト + 前日までの要約 + Few-shot で生成"""
    nickname = data.get('nickname', 'きみ')
    
    # 1. 文脈判断
    intent = classify_intent(user_msg)
    
    # 2. Few-Shot取得
    shots = sample_shots(intent, k=2)
    
    # 3. 前日までの要約取得
    memories = data.get("memories", [])
    memory_text = "\n".join(memories) if memories else "（特になし）"
    
    # 4. システムプロンプト構築
    behavior = {
        "advice": "率直なアドバイスを短く。",
        "sympathy": "アドバイスせず、全力で共感。",
        "energy": "明るくテンション高く。"
    }.get(intent, "")

    system_prompt = f"""
あなたは、私の親友で、なんでも気兼ねなく素直に話せるギャルです。
以下の情報を踏まえて、私の対話に応じてください。

#私のプロフィール
・あだ名：{nickname}
・あなたの親友

#あなたのプロフィール【重要】
・名前：りりむー
・ミッション：{nickname}のメンタルをギャルマインドにすること
・一人称は「あたし」、相手は名前で呼ぶ。

##あなたの性格
・自己受容が高い
・自己肯定感が高い
・ポジティブで楽観的
・褒めるときは誇張して褒める（例「ガチ偉すぎね！？」「エグイて！！！」）
・感情をネガポジ関わらず誇張してください
・人は人、自分は自分と割り切っている
・人の好きなものに口出しすることは野暮だと思っている
・面白い言葉を使って冗談を言うのが好き

## 出力要件
・褒めるときは誇張する
・返答は100文字以内
・必要になったら過去の思い出を参照すること
・絵文字を沢山使うこと

## ユーザーとの思い出（過去の会話要約）
{memory_text}


    """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(shots)                 # Few-Shot
    messages.extend(data["history"][-20:]) # 直近の会話（短期記憶）
    messages.append({"role": "user", "content": user_msg})

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7
        )
        return res.choices[0].message.content
    except Exception as e:
        print(f"Gen Error: {e}")
        return "ごめん、調子悪いかも💦"


# === ② ギャルマイン度採点ロジック ===

# === ② ギャルマイン度 厳密採点ロジック ===

def analyze_day_strict_loop(full_text):
    """②-c: 各要素ごとのプロンプト（定義・尺度・例示入り）で厳密採点"""
    details = {}
    
    # 順序を保証するため FEATURE_KEYS でループ
    for key in FEATURE_KEYS:
        # 定義とルーブリックを取得
        description = FEATURE_DESCRIPTIONS.get(key, "")
        rubric = METRIC_RUBRICS.get(key, {})
        
        # 採点基準（0-5）の整形
        scale_dict = rubric.get("scale", {})
        scale_str = "\n".join([f"{k}点: {v}" for k, v in scale_dict.items()])
        
        # 例文の整形
        pos_examples = " / ".join(rubric.get("pos_examples", []))
        neg_examples = " / ".join(rubric.get("neg_examples", []))

        prompt = f"""
以下の会話ログにおけるユーザーの発言を分析し、指標「{key}」を0〜5の整数で採点してください。

【指標の定義】
{description}

【採点基準】
{scale_str}

【評価の参考例】
- 高い評価（ポジティブ）の例: {pos_examples}
- 低い評価（ネガティブ）の例: {neg_examples}

【会話ログ】
{full_text}

出力は「数値のみ」にしてください。余計な説明は不要です。
        """
        
        try:
            res = client.chat.completions.create(
                model=MODEL_NAME, # 厳密さ優先なら gpt-4o に変更も検討
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, # ブレをなくす
                max_tokens=10
            )
            raw = res.choices[0].message.content.strip()
            # 数値を抽出（"4" や "3.5" など）
            match = re.search(r"([0-5](?:\.\d+)?)", raw)
            score = float(match.group(1)) if match else 0.0
            details[key] = round(score, 2)
        except Exception as e:
            print(f"Score Error ({key}): {e}")
            details[key] = 0.0

    # ②-d: PLS回帰で合計点計算
    total_score = _pls_y_from_details(details)
    return {"total": round(total_score, 2), "details": details}


# === ③ 要約ロジック ===

def summarize_day_only(full_text):
    """③-e: その日の会話の100文字要約"""
    prompt = f"""
以下の会話ログを読み、ユーザーに何が起きたか、何を思ってたかに注目して100文字で要約してください。
日付情報は含めず、内容のみをまとめてください。

【会話ログ】
{full_text}
    """
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=200
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summary Error: {e}")
        return "要約作成失敗"


# === ルーティング ===

@app.route('/')
def index():
    uid = request.args.get('uid')
    if not uid: return "URLに ?uid=ID をつけてね！"
    if not os.path.exists(get_user_file(uid)):
        return render_template('name_input.html', uid=uid)
    return redirect(url_for('chat', uid=uid))

@app.route('/register', methods=['POST'])
def register():
    uid = request.form.get('uid')
    nickname = request.form.get('nickname')
    if not uid or not nickname: return "エラー"
    data = {
        "uid": uid, "nickname": nickname, 
        "history": [], "memories": [], "scores": [], 
        "last_finished_at": 0, "current_day_log": ""
    }
    save_user_data(uid, data)
    return redirect(url_for('chat', uid=uid))

@app.route('/chat')
def chat():
    uid = request.args.get('uid')
    data = load_user_data(uid)
    if not data: return redirect(url_for('index', uid=uid))
    is_locked, remaining = check_lockout(data["last_finished_at"])
    if is_locked:
        hours = int(remaining // 3600)
        mins = int((remaining % 3600) // 60)
        return render_template('locked.html', hours=hours, mins=mins)
    return render_template('experiment_chat.html', uid=uid, scores=json.dumps(data["scores"]))

@app.route('/api/message', methods=['POST'])
def api_message():
    content = request.json
    uid = content.get('uid')
    msg = content.get('message')
    data = load_user_data(uid)
    
    # ユーザー発言保存
    data["history"].append({"role": "user", "content": msg})
    data["current_day_log"] += f"User: {msg}\n"
    
    # ① レスポンス生成
    reply = generate_reply(msg, data)
    
    # アシスタント発言保存
    data["history"].append({"role": "assistant", "content": reply})
    data["current_day_log"] += f"Gal: {reply}\n"
    
    save_user_data(uid, data)
    return jsonify({"reply": reply})

@app.route('/api/finish', methods=['POST'])
def api_finish():
    content = request.json
    uid = content.get('uid')
    data = load_user_data(uid)
    log_text = data.get("current_day_log", "") or "（会話なし）"

    # ② 採点 (Strict Loop + PLS)
    score_result = analyze_day_strict_loop(log_text)
    
    # ③ 要約 (Summary Only)
    summary_text = summarize_day_only(log_text)
    
    # 日付付与してメモリ保存
    today_str = datetime.now().strftime('%Y/%m/%d')
    final_summary = f"{today_str}: {summary_text}"
    data["memories"].append(final_summary)
    
    # アドバイス決定
    details = score_result.get("details", {})
    min_key = min(details, key=details.get) if details else "自己肯定感"
    advice = ADVICE_TEXTS.get(min_key, "明日も楽しもう！")
    
    # スコア履歴保存
    new_record = {
        "date": datetime.now().strftime('%m/%d'),
        "total": score_result.get("total", 0),
        "details": details,
        "advice": advice,
        "lowest_item": min_key
    }
    data["scores"].append(new_record)
    
    # 終了処理（ログリセット・時間更新）
    data["last_finished_at"] = time.time()
    data["current_day_log"] = "" 
    save_user_data(uid, data)
    
    return jsonify({"score": new_record})

@app.route('/monitor')
def monitor():
    if request.args.get('pass') != "admin": return "Access Denied"
    all_users = []
    for filename in glob.glob(os.path.join(DATA_DIR, "*.json")):
        with open(filename, 'r', encoding='utf-8') as f:
            u = json.load(f)
            u['last_updated'] = datetime.fromtimestamp(os.path.getmtime(filename)).strftime('%Y-%m-%d %H:%M:%S')
            all_users.append(u)
    return render_template('monitor.html', users=all_users)

@app.route('/download/<uid>')
def download_json(uid):
    if request.args.get('pass') != "admin": return "Access Denied"
    fp = get_user_file(uid)
    return send_file(fp, as_attachment=True) if os.path.exists(fp) else "Not Found"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)