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
AI_MODEL_NAME = "gpt-5.4" 

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

def format_shots(shots: List[Dict]) -> str:
    formatted = []
    for shot in shots:
        role = shot.get("role")
        content = shot.get("content", "")
        if role == "user":
            formatted.append(f"User: {content}")
        elif role == "assistant":
            formatted.append(f"Assistant: {content}")
    return "\n".join(formatted)

# ==========================
# Scoring Class
# ==========================
class Scoring:
    """
    - 直前の履歴（ユーザ発話のみ）を「文字数」で窓切り（例: 50文字）
    - 8項目を項目別プロンプトで0〜5採点
    - PLS(標準化β)で合成 → 0〜50クリップ（UI想定）
    """
    def __init__(self, client: OpenAI, model: str = "gpt-4.1-mini", window_chars: int = 50):
        self.client = client
        self.model = model
        self.window_chars = window_chars

        # 表示用項目と説明（プロンプトで使う）
        self.metric_prompts = {
            "自己肯定感": "自分をどれだけ「良い/価値がある」と評価しているか",
            "自己受容": "長所も短所も含め、あるがままの自分を受け入れる姿勢",
            "楽観性": "物事がうまく進むと一般的に期待する傾向",
            "自他境界": "感情に巻き込まれず・同一化/遮断に偏らず立場を保てる傾向",
            "本来性": "外圧に過度に左右されず価値観に沿って選ぶ傾向",
            "他者尊重": "相手の価値・個性・尊厳を尊重する態度",
            "感情の強度": "強調語/感嘆の多さなど感情表出の強さ",
            "言語創造性": "スラング/造語/比喩等の創造的表現",
        }
        # Scoring.__init__ 内
        self.metric_rubrics = {
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

    def _recent_user_chars(self, history: List[Dict], user_response: str) -> str:
        """
        直前履歴（ユーザ発話のみ）と今回の user_response を結合し、
        後ろから window_chars 文字を抜く
        """
        texts = [m["content"] for m in history if m.get("role") == "user"] + [user_response]
        s = "".join(texts)
        if len(s) <= self.window_chars:
            return s
        return s[-self.window_chars:]

    def _build_metric_prompt(self, display_name: str, hint: str, context_text: str) -> str:
        rubric = getattr(self, "metric_rubrics", {}).get(display_name)
        parts = [
            f"次の日本語テキストを読み、指標「{display_name}」を0〜5点で採点してください。",
            f"【指標説明】{hint}",
        ]
        if rubric:
            scale_lines = " / ".join(f"{k}:{v}" for k, v in rubric["scale"].items())
            pos = "」「".join(rubric.get("pos_examples", []))
            neg = "」「".join(rubric.get("neg_examples", []))
            parts.append(f"【採点基準】{scale_lines}")
            if pos:
                parts.append(f"【高い表現の例】「{pos}」")
            if neg:
                parts.append(f"【低い表現の例】「{neg}」")
        parts += [
            f"【文脈】{context_text}",
            "出力は数値のみ（0〜5、小数点可）。余計な文字は出さない。",
        ]
        return "\n".join(parts)

    def score_from_context(self, context_text: str) -> dict:
        display_scores: Dict[str, float] = {}
        for disp, hint in self.metric_prompts.items():
            prompt = self._build_metric_prompt(display_name=disp, hint=hint, context_text=context_text)
            try:
                res = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_completion_tokens=6,
                )
                raw = (res.choices[0].message.content or "").strip()
                m = re.search(r"([0-5](?:\\.\\d+)?)", raw)
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

    def score(self, history: List[Dict], user_response: str) -> Dict:
        """
        戻り値：
        {
          "total": float,                 # 0-50
          "details_display": {表示名:score},     # 0-5
          "details_model":   {学習名:score},     # 0-5（PLS合成用）
          "context_excerpt": str,         # 使用した抜粋
          "window_chars": int
        }
        """
        context_text = self._recent_user_chars(history, user_response)
        return self.score_from_context(context_text)

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
                temperature=0.0, max_completion_tokens=10,
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
            "要件: 回答は100文字以内。絵文字をたくさん使ってください。\n"
            "【注意】一般倫理的に違反することや犯罪を助長することは絶対に書かないでください。また、性的な内容も避けてください."


        )
        
        shots_text = format_shots(shots)
        enhanced_system = self.system_prompt + "\n\nこれらの対話例は、応答の際の口調の適応に参考にすること:\n" + shots_text
        
        messages = [{"role": "system", "content": enhanced_system}]
        messages.append({"role": "user", "content": f"{style_instruction}\n\nUser: {user_response}"})

        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=REPLY_TOKENS,
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
    scorer = Scoring(client, model=AI_MODEL_NAME, window_chars=50)
    history_without_current = sess["history"][:-1]
    result = scorer.score(history_without_current, user_msg)

    # セッションデータ更新
    sess["gmd_totals"].append(result["total"])
    sess["gmd_details"].append(result["details_display"])
    sess["eval_count"] += 1

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
