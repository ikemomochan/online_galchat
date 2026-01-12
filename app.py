import os, re, uuid, time, hashlib, random
from datetime import timedelta
from typing import List, Dict, Optional
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 設定関連 ──
CHAT_MODE = "chat"          
LISTENING_MODE = False      
MIN_TOKENS = 50             
REPLY_TOKENS = 150          

# モデル名
AI_MODEL_NAME = "gpt-4.1-mini" 

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "reset_2025_11_29_v2" 
app.permanent_session_lifetime = timedelta(days=365) 
CORS(app, resources={r"/ask": {"origins": "*"}})

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 開発者用パスワード
DEV_PASSWORD = "admin"

# 制限時間（秒）: 5分 = 300秒
TIME_LIMIT_SECONDS = None

# BANリスト
BANNED_USERS = set()

# セッション管理
SESSIONS: Dict[str, Dict] = {}

# ── ヘルパー関数 ──

def get_session(sid: str) -> Dict:
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "history": [],
            "eval_count": 0,
            "gmd_totals": [],
            "gmd_details": [],
            "scoring_buffer": "", 
        }
    return SESSIONS[sid]

def is_banned():
    if session.get('is_dev'):
        return False
    user_ip = request.remote_addr
    if user_ip in BANNED_USERS or session.get('is_finished'):
        return True
    return False

FEATURE_KEYS = [
    "自己肯定感", "自己受容", "楽観性", "自他境界",
    "本来性", "他者尊重", "感情の強度", "言語創造性",
]
FEATURE_KEYS_MODEL = FEATURE_KEYS

# PLS(標準化β) 設定
CONST = {
    "std_beta": [1.106135, 1.370707, 0.874889, -0.04322, 0.477184, 0.133321, 1.535123, 1.133268],
    "mu_y": 26.61,
    "sigma_y": 3.89,
    "mu_xs": [3.39, 3.51, 3.45, 3.72, 3.77, 3.90, 2.87, 1.06],
    "sigma_xs": [0.62, 0.58, 0.65, 0.59, 0.69, 0.67, 0.89, 0.80],
}

DISPLAY2MODEL = {k: k for k in FEATURE_KEYS}

SYSTEM_PROMPT = (
    """
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
    """
)

LISTENING_SYSTEM_PROMPT = """
あなたは落ち着いた聞き役AIです。
相手の気持ちを否定せず、共感やうなずきを示してください。
文章は80文字程度までの短いまとまりで返してください。
"""

INTENT_LABELS = ["advice", "sympathy", "energy"]

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

def build_system_prompt(profile: Optional[Dict], mode: str) -> str:
    base = LISTENING_SYSTEM_PROMPT if mode in {"listen", "listening", "listener"} else SYSTEM_PROMPT
    if profile:
        return base.format(nickname=profile.get('nickname','きみ'))
    return base.format(nickname='きみ')

def sample_shots(intent: str, k: int = 2, sid: Optional[str] = None) -> List[Dict]:
    pool = FEW_SHOTS.get(intent, FEW_SHOTS.get("energy", []))
    pairs = [(pool[i], pool[i+1]) for i in range(0, len(pool), 2) if i+1 < len(pool)]
    if not pairs: return []
    
    seed = int(hashlib.md5((sid or "anon").encode()).hexdigest(), 16) & 0xffffffff
    rnd = random.Random(seed)
    picked = pairs[:max(0, min(k, len(pairs)))]
    return [m for p in picked for m in p]

def _std_vec(x_vals: List[float]) -> np.ndarray:
    mu = np.array(CONST["mu_xs"], float)
    sd = np.array(CONST["sigma_xs"], float)
    x  = np.array(x_vals, float)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd

def _pls_y_from_details(details_model_keys: Dict[str, float]) -> float:
    x = [float(details_model_keys.get(k, 0.0)) for k in FEATURE_KEYS_MODEL]
    z = _std_vec(x)
    beta = np.array(CONST["std_beta"], float)
    z_pred = float(z @ beta)
    y = float(CONST["mu_y"] + CONST["sigma_y"] * z_pred)
    return y

def _clip_0_50(v: float) -> float:
    return max(0.0, min(50.0, v))

def _bubble_split(text: str, max_bubbles: int = 3) -> List[str]:
    if not text: return [""]
    s = re.sub(r"\s+", " ", text).strip()
    chunks = re.findall(r".+?(?:[。．！？!?]+|$)", s)
    if not chunks: chunks = [s]
    return chunks[:max_bubbles]

# ==========================
# Scoring Class
# ==========================
class Scoring:
    def __init__(self, client: OpenAI, model: str, window_chars: int = 50):
        self.client = client
        self.model = model
        self.window_chars = window_chars
        self.metric_prompts = {
            "自己肯定感": "自分をどれだけ「良い/価値がある」と評価しているか",
            "自己受容": "長所も短所も含め、あるがままの自分を受け入れる姿勢",
            "楽観性": "物事がうまく進むと一般的に期待する傾向",
            "自他境界": "感情に巻き込まれず立場を保てる傾向",
            "本来性": "価値観に沿って選ぶ傾向",
            "他者尊重": "相手の価値・個性を尊重する態度",
            "感情の強度": "感情表出の強さ",
            "言語創造性": "スラング/造語/比喩等の創造的表現",
        }

    def score_from_context(self, context_text: str) -> dict:
        display_scores: Dict[str, float] = {}
        for disp, hint in self.metric_prompts.items():
            prompt = f"テキスト:「{context_text}」\n指標: {disp}({hint})\nこの指標を0〜5点で評価し、数値のみ出力せよ。"
            try:
                res = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=5,
                )
                raw = (res.choices[0].message.content or "").strip()
                m = re.search(r"([0-5](?:\.\d+)?)", raw)
                score = float(m.group(1)) if m else 0.0
            except Exception:
                score = 0.0
            display_scores[disp] = round(score, 2)

        model_scores = {DISPLAY2MODEL.get(d, d): float(v) for d, v in display_scores.items()}
        raw_y = _pls_y_from_details(model_scores)
        total = _clip_0_50(raw_y)

        return {
            "total": round(total, 2),
            "details_display": display_scores,
            "details_model": model_scores,
            "context_excerpt": context_text,
            "window_chars": len(context_text),
        }

# ==========================
# Response Class
# ==========================
class Response:
    def __init__(self, client: OpenAI, model: str, system_prompt: str, profile: Optional[Dict], mode: str):
        self.client = client
        self.model = model
        self.mode = mode
        self.system_prompt = build_system_prompt(profile, self.mode)

    def classify_intent(self, user_response: str) -> str:
        if self.mode in {"listen", "listening"}: return "listen"
        system = "ユーザー入力が advice, sympathy, energy のどれに該当するか1語で出力せよ。"
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system},{"role":"user","content":user_response}],
                temperature=0.0, max_tokens=10,
            )
            raw = (res.choices[0].message.content or "").strip().lower()
            return next((label for label in INTENT_LABELS if label in raw), "energy")
        except:
            return "energy"

    def plan_target(self, current_total: Optional[float]) -> float:
        return _clip_0_50((current_total or 30.0) + 5.0)

    def generate_reply(self, user_response: str, intent: str, target_score: float, max_bubbles: int, profile: Optional[Dict], mode: str, sid: str) -> List[str]:
        shots = sample_shots(intent, k=2, sid=sid)
        
        style_instruction = (
            f"Intent: {intent}\nTarget Score: {target_score}\n"
            "要件: 回答は40文字以内。絵文字をたくさん使ってください。\n"
            "【注意】一般倫理的に違反することや犯罪を助長することは絶対に書かないでください。また、性的な内容も避けてください."


        )
        
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(shots)
        messages.append({"role": "user", "content": f"{style_instruction}\n\nUser: {user_response}"})

        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=REPLY_TOKENS,
            )
            text = (res.choices[0].message.content or "").strip()
        except Exception as e:
            text = f"ごめん、今ちょいエラー出たかも💦 ({e})"

        return _bubble_split(text, max_bubbles)

# ── ルーティング ──

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if is_banned():
        return render_template('game_over.html', message="体験版の利用は終了しました！ありがとね💖")

    if request.method == 'POST':
        nickname = request.form.get('nickname')
        session.permanent = True
        session['profile'] = {'nickname': nickname}
        
        if nickname == DEV_PASSWORD:
            session['is_dev'] = True
            session['profile']['nickname'] = "開発者ちゃん"
        else:
            session['is_dev'] = False

        if 'start_time' not in session:
            session['start_time'] = time.time()
            
        return redirect(url_for('chat'))
    return render_template('profile.html')

@app.route('/chat')
def chat():
    if is_banned():
        return render_template('game_over.html', message="体験版の利用は終了しました！ありがとね💖")
    if 'profile' not in session:
        return redirect(url_for('profile'))
    return render_template('gal_index.html', profile=session['profile'])

@app.route("/")
def index():
    return redirect(url_for('profile'))

@app.route("/ask", methods=["POST"])
def ask():
    # 1. BANチェック
    if is_banned():
        return jsonify({"error": "Time limit exceeded", "answer": "もう終了時間だよ～！またね👋", "force_stop": True}), 403
    # 2. No time limit
    remaining_val = None

    payload = request.json or {}
    sid = request.cookies.get("sid") or payload.get("sid") or str(uuid.uuid4())
    sess = get_session(sid)
    user_profile = session.get('profile')
    user_msg = (payload.get("message") or "").strip()

    if not user_msg:
        return jsonify({"sid": sid, "answer": "え、なんて？💦"})

    sess["history"].append({"role": "user", "content": user_msg})

    # ★★★ 新しい採点ロジック ★★★
    # 1. バッファに今回の発言を追記
    current_buffer = sess.get("scoring_buffer", "")
    current_buffer += user_msg
    sess["scoring_buffer"] = current_buffer
    
    # 2. バッファの長さが50文字を超えているかチェック
    score_result = None
    if len(current_buffer) >= 50:
        # 採点
        scorer = Scoring(client, model=AI_MODEL_NAME, window_chars=len(current_buffer))
        result = scorer.score_from_context(current_buffer)
        
        # セッションデータ更新
        sess["gmd_totals"].append(result["total"])
        sess["gmd_details"].append(result["details_display"])
        sess["eval_count"] += 1
        
        # バッファを空にする
        sess["scoring_buffer"] = ""
        
        # フロントエンドへの返却用データ作成
        score_result = {
            "total": result["total"],
            "details": result["details_display"],
            "eval_index": sess["eval_count"] - 1,
        }

    # AI返答生成
    responder = Response(client, model=AI_MODEL_NAME, system_prompt=SYSTEM_PROMPT, profile=user_profile, mode=CHAT_MODE)
    intent = responder.classify_intent(user_msg)
    target = responder.plan_target(sess["gmd_totals"][-1] if sess.get("gmd_totals") else None)
    
    bubbles = responder.generate_reply(
        user_response=user_msg,
        intent=intent,
        target_score=target,
        max_bubbles=2,
        profile=user_profile,
        mode=CHAT_MODE,
        sid=sid
    )

    if bubbles:
        sess["history"].append({"role": "assistant", "content": bubbles[0]})

    # レスポンス作成
    resp_payload = {
        "sid": sid, 
        "answer": bubbles, 
        "intent": intent, 
        "remaining_seconds": remaining_val
    }

    # ★★★ ここが抜けていました！ ★★★
    # 計算したスコア結果があるなら、返却データに追加する
    if score_result:
        resp_payload["gmd"] = score_result

    response = jsonify(resp_payload)
    response.set_cookie("sid", sid, max_age=60*60*24*30, httponly=True, samesite="Lax")
    return response, 200

# 救済用URL
@app.route('/reset_test') 
def reset_test():
    user_ip = request.remote_addr
    if user_ip in BANNED_USERS:
        BANNED_USERS.remove(user_ip)
    session.clear()
    return "リセット完了！ /profile に戻って、また 'admin' で入ってね💖"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)