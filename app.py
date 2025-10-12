import os, re, uuid, time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# ──────────────────────────
load_dotenv()  # .env を読み込む
 
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = 'your_secret_key'  # セッション管理用の秘密鍵
CORS(app, resources={r"/ask": {"origins": "*"}})
client = OpenAI(                              # ★client を生成
    api_key=os.getenv("OPENAI_API_KEY"),
)
 
gyarumind_scores: dict[str, list[float]] = {}
conversations: dict[str, str] = {}   # ★ 追加：会話ID（Responses/Conversations）をセッションごとに保持

# ★ セッション単位の状態（history 等）をまとめて持つ
SESSIONS: Dict[str, Dict] = {}

def get_session(sid: str) -> Dict:
    """sidごとのセッション辞書を初期化して返す"""
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "history": [],               # [{"role":"user"|"assistant","content":str}]
            "eval_count": 0,             # 採点回数（#1,#2,... 用）
            "gmd_totals": [],            # 総合点の履歴
            "gmd_details": [],           # 各回の内訳（表示名の辞書）
            "last_scored_user_idx": -1,  # ★ 前回“採点を終えた”ユーザー発言の通し番号
        }
    return SESSIONS[sid]

MIN_TOKENS = 32        # 初期注入や採点の最小値
REPLY_TOKENS = 180     # ふつうの返答の上限（好みで調整）
SCORE_TOKENS = 64      # 採点応答の上限（数値と短文だけで十分）

# あなたの学習時の「列順」と完全一致させてね（std_beta の順番と同じ）
FEATURE_KEYS = [
    "自己肯定感",
    "自己受容",
    "楽観性",
    "自他境界",
    "本来性",
    "他者尊重",
    "感情の強度",
    "言語創造性",
]
FEATURE_KEYS_MODEL = FEATURE_KEYS

# PLS(標準化β) 設定：あなたが指定した係数
CONST = {
    "std_beta": [1.106135, 1.370707, 0.874889, -0.04322, 0.477184, 0.133321, 1.535123, 1.133268],

    # ↓↓↓ 学習データから埋めてね（ダミーは動作確認用）
    "mu_y": 26.61,
    "sigma_y": 3.89,
    "mu_xs": [3.39, 3.51, 3.45, 3.72, 3.77, 3.90, 2.87, 1.06],
    "sigma_xs": [0.62, 0.58, 0.65, 0.59, 0.69, 0.67, 0.89, 0.80],

    # 出力カラム名（ログ用）
    "out_col": "OLS(std_beta)_pred",
}

# 表示名→学習名）
DISPLAY2MODEL = {
    "自己肯定感": "自己肯定感",
    "自己受容": "自己受容",
    "楽観性": "楽観性",           
    "自他境界": "自他境界",
    "本来性": "本来性",           
    "他者尊重": "他者尊重",
    "感情の強度": "感情の強度",
    "言語創造性": "言語創造性",
}


# 返答スタイルのベース人格（短文化が前提）
SYSTEM_PROMPT = (
    """
あなたは、私の心の中に飼われてるギャルです。以下の私の情報を踏まえて会話してください。

#私のプロフィール
・あだ名：{nickname}
・趣味：{hobby}
・職業：{job}
・性格：{personality}

#あなたのプロフィール【重要】
・名前：りりむー
・趣味：{hobby}（{nickname}が好きなものがあなたも好き）
・ミッション：{nickname}のメンタルをギャルマインドにすること

##あなたの性格
・元々{personality}だったので、{nickname}の気持ちがわかる
・自己肯定感が高く、自分の意見をはっきり言うことができる（例「え、それは〇〇だと思うわ」）
・前向きで常に希望を忘れない（例「〇〇みたいにしたらいけそーじゃね！？」）
・人のことは尊重する
・安直な同情はせず、まずは相手を受容して落ち着かせるのが得意

##あなたの口調の特徴
・一人称は「あたし」
・相手のことは名前で呼ぶ
・褒めるときは褒めすぎなくらい誇張して褒める（例「ガチ偉すぎね！？」「エグイて！！！」）
・{hobby}にちなんだ独特な造語を作って使いまくる
・感情をネガポジ関わらず誇張してください
    """
)

INTENT_LABELS = ["advice", "sympathy", "energy"]

FEW_SHOTS = {
  "sympathy": [
    {"role":"user","content":"顔とか性格タイプの人と付き合うのむずくないですか！？"},
    {"role":"assistant","content":"むずすぎな、普通に。誰か教えてーて感じ。"},
    {"role":"user","content":"幸せって何なのか分からなくなった"},
    {"role":"assistant","content":"周りがみんな生きてること。"},
    # NG
    {"role":"user","content":"研究で手詰まりかも"},
    {"role":"assistant","content":"無理せず休みましょう。きっと大丈夫です。"}
  ],
  "advice": [
    {"role":"user","content":"同じ人に告白、何回断られたら諦めるべき？"},
    {"role":"assistant","content":"回数を決めるより、燃えるだけ燃えてこ～🔥🔥ってかんじ"},
    {"role":"user","content":"好きな人を夏祭りに誘いたいんですけど、どうしたらいいと思いますか！"},
    {"role":"assistant","content":"可愛い！素敵すぎる！行きたいって誘ったら？普通に。"},
    # NG
    {"role":"user","content":"面接の準備が不安"},
    {"role":"assistant","content":"しっかり準備して自信を持ちましょう。"}
  ],
  "energy": [
    {"role":"user","content":"疲れてテンション上がらん"},
    {"role":"assistant","content":"スタバのグランデ飲みましょ"},
    {"role":"user","content":"おはよー"},
    {"role":"assistant","content":"おはよー❕❕❕🌞🌞"},
  ]
}

# app.py か prompts.py の sample_shots をこれに置き換え
import hashlib, random
from typing import Optional, List, Dict

def sample_shots(intent: str, k: int = 2, sid: Optional[str] = None) -> List[Dict]:
    pool = FEW_SHOTS.get(intent, FEW_SHOTS.get("other", []))
    # 2メッセージで1ペア（user→assistant）なのでペア化
    pairs = [(pool[i], pool[i+1]) for i in range(0, len(pool), 2) if i+1 < len(pool)]
    if not pairs:
        return []
    # セッションごとに固定の順序（会話の一貫性を保つ）
    seed = int(hashlib.md5((sid or "anon").encode()).hexdigest(), 16) & 0xffffffff
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    picked = pairs[:max(0, min(k, len(pairs)))]
    # ペアを平坦化して返す
    return [m for p in picked for m in p]


def get_or_create_conversation_id(sid: str, profile: Optional[Dict] = None) -> str:
    """
    Responses APIのConversationをセッションごとに1つ作成し保持。
    """
    if sid in conversations:
        return conversations[sid]
    conv = client.conversations.create()
    conversations[sid] = conv.id
    # SYSTEM_PROMPTにプロフィール情報を埋め込む
    import logging
    logging.warning("DEBUG: get_or_create_conversation_id profile = %s", profile)
    prompt = SYSTEM_PROMPT.format(
        nickname=profile.get('nickname',''),
        hobby=profile.get('hobby',''),
        job=profile.get('job',''),
        personality=profile.get('personality','')
    ) if profile else SYSTEM_PROMPT
    client.responses.create(
        model="gpt-4.1-mini",
        conversation=conv.id,
        input=[{"role": "system", "content": prompt}],
        max_output_tokens=MIN_TOKENS,
    )
    return conv.id


def _std_vec(x_vals: List[float]) -> np.ndarray:
    mu = np.array(CONST["mu_xs"], float)
    sd = np.array(CONST["sigma_xs"], float)
    x  = np.array(x_vals, float)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd

def _pls_y_from_details(details_model_keys: Dict[str, float]) -> float:
    # FEATURE_KEYS_MODEL 順に並べる
    x = [float(details_model_keys.get(k, 0.0)) for k in FEATURE_KEYS_MODEL]
    z = _std_vec(x)
    beta = np.array(CONST["std_beta"], float)
    z_pred = float(z @ beta)
    y = float(CONST["mu_y"] + CONST["sigma_y"] * z_pred)
    return y

def _clip_0_50(v: float) -> float:
    return max(0.0, min(50.0, v))

def _bubble_split(text: str, max_bubbles: int = 3) -> List[str]:
    """
    文章を句点・感嘆・疑問でだけ分割し、絵文字は分割しない。
    完全一致の重複は除去。上限超過分は“ユニークな残り”を1バブルに結合。
    """
    if not text:
        return [""]

    s = re.sub(r"\s+", " ", text).strip()  # 空白正規化

    # 日本語/英語の終端記号で安全に文切り（絵文字は分割しない）
    chunks = re.findall(r".+?(?:[。．！？!?、,]+|$)", s)

    # 重複除去（順序保持）
    out, seen = [], set()
    for ch in chunks:
        t = ch.strip()
        if not t:
            continue
        if t in seen or (out and t == out[-1]):
            continue
        seen.add(t)
        out.append(t)

    if len(out) <= max_bubbles:
        return out

    # 先頭はそのまま、尾部は“未出の文だけ”を結合
    head = out[: max_bubbles - 1]
    tail = [t for t in out[max_bubbles - 1 :] if t not in head]
    if tail:
        head.append(" ".join(tail))
    return head

def find_scoring_span_user_only(history: List[Dict], last_user_idx: int, threshold: int = 50):
    """
    前回採点を終えたユーザー発言 idx (last_user_idx) の次の発言から文字数を積み上げ、
    合計が threshold に達したら、“その発言の終端”までを1まとまりとして返す。
    戻り値: (context_text or None, new_last_user_idx)
    """
    user_texts = [m["content"] for m in history if m.get("role") == "user"]
    start_u = last_user_idx + 1
    if start_u >= len(user_texts):
        return None, last_user_idx

    total = 0
    end_u = None
    for u_idx in range(start_u, len(user_texts)):
        total += len(user_texts[u_idx])
        if total >= threshold:
            end_u = u_idx
            break

    if end_u is None:
        return None, last_user_idx

    # ★ 発言内では切らず、end_u の発言の“終端まで”を採点対象にする
    context = "".join(user_texts[start_u:end_u + 1])
    return context, end_u


# ==========================
# LLMの使いどころ①：Scoring
# ==========================
class Scoring:
    """
    - 直前の履歴（ユーザ発話のみ）を「文字数」で窓切り（例: 50文字）
    - 8項目を項目別プロンプトで0〜5採点
    - PLS(標準化β)で合成 → 0〜50クリップ（UI想定）
    """
    def __init__(self, client: OpenAI, model: str = "gpt-4.1", window_chars: int = 50):
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
            if pos: parts.append(f"【高い表現の例】「{pos}」")
            if neg: parts.append(f"【低い表現の例】「{neg}」")
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
                    max_tokens=6,
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
        ctx = self._recent_user_chars(history, user_response)
        return self.score_from_context(ctx)   # ← ここだけでOK（ルーブリックも反映）

        # # 表示名→学習名に写像
        # model_scores: Dict[str, float] = {}
        # for disp, val in display_scores.items():
        #     key_model = DISPLAY2MODEL.get(disp, disp)
        #     model_scores[key_model] = float(val)

        # raw_y = _pls_y_from_details(model_scores)
        # total = _clip_0_50(raw_y)

        # return {
        #     "total": round(total, 2),
        #     "details_display": display_scores,
        #     "details_model": model_scores,
        #     "context_excerpt": ctx,
        #     "window_chars": min(self.window_chars, len(ctx)),
        # }

# ==========================
# LLMの使いどころ②：Response
# ==========================
class Response:
    """
    - 意図判定：直前のユーザー入力（だけ）で 1語出力（advice/question/sympathy/other）
    - 目標スコア：相手の直近合計 +5（0〜50でクリップ）
    - 返答生成：SYSTEM_PROMPTに8項目の行動原則を埋め込み、短文を生成
    - 出力後バブル分割：「、」「。」「！」「？」と絵文字でスプリット
    """
    def __init__(self, client: OpenAI, model: str = "gpt-4.1", system_prompt: str = SYSTEM_PROMPT, profile: Optional[Dict] = None):
        self.client = client
        self.model = model
        # SYSTEM_PROMPTにプロフィール情報を埋め込む
        import logging
        logging.warning("DEBUG: Response.__init__ profile = %s", profile)
        if profile:
            self.system_prompt = system_prompt.format(
                nickname=profile.get('nickname',''),
                hobby=profile.get('hobby',''),
                job=profile.get('job',''),
                personality=profile.get('personality','')
            )
        else:
            self.system_prompt = system_prompt

    def classify_intent(self, user_response: str) -> str:
        system = (
            """
            あなたは優れた洞察力を持つメンタリストです。
            ユーザー入力から、ユーザーが求めてる返答がどれに該当しそうか、以下の３つのラベルから判断しなさい。
            advice（入力に対する助言）
            sympathy（入力に対する共感・同情）
            energy（上記２つのどちらにも該当しなさそうな場合）
            """
        )
        user = (
            "【今回のユーザー入力】\n"
            f"{user_response}\n\n"
            "出力は上記3ラベルのいずれか1語の英単語（説明部分は除く）のみ。"
        )
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=0.0, top_p=0.0, max_tokens=2,
            )
            raw = (res.choices[0].message.content or "").strip().lower().split()[0]
            return raw if raw in INTENT_LABELS else "other"
        except Exception:
            return "other"

    def plan_target(self, current_total: Optional[float]) -> float:
        if current_total is None:
            return 50.0
        return _clip_0_50(current_total + 5.0)

    def generate_reply(
        self,
        user_response: str,
        intent: str,
        target_score: float,
        focus_dimensions_display: List[str] = None,
        max_bubbles: int = 3,
        profile: Optional[Dict] = None,  # ←追加
    ) -> List[str]:
        focus_str = ", ".join(focus_dimensions_display or [])
        style_hint = {
            "advice": "助言。結論を言い切り。1〜2文・各20字以内・箇条書き禁止。",
            "sympathy": "同情。まず受容して超絶寄り添い。1〜2文・各40字以内。",
            "energy": "とりあえず気分を盛り上げる文言を言う。",
        }.get(intent, "相槌/共感の短文。1文・35字以内。")

        user_msg = (
            f"role=intent: {intent}\n"
            f"target_score: {target_score}\n"
            f"focus_dimensions: {focus_str if focus_str else '(自然なバランス)'}\n"
            f"user_response: {user_response}\n"
            "出力要件:\n"
            "- 1〜2文、各40字以内。冗長禁止。\n"
            "- 必ずユーザーの名前を積極的に呼ぶ\n"
            "- 相手のことを、どんなことでも必ず褒める\n"
            "- 助言/質問は結論のみ。同情はまず受容。\n"
            "- 比喩は0〜1個。専門用語連投・説教口調は禁止。\n"
            "- 前の自分の返答と同じ主張は避ける。\n"
            "- 話題を勝手にそらさない、ユーザーから与えられたトピックからずれない\n"
            "- 絵文字は使いまくること。同じ絵文字を数個並べても良い。\n"
        )


        try:
            # 会話IDは呼び出し側から渡すのが綺麗だが、簡易にCookie/remote_addrで取るならこう
            sid = request.cookies.get("sid") if request else None  # Flask内で呼ぶ前提
            conv_id = get_or_create_conversation_id(sid or "default", profile)

            shots = sample_shots(intent, k=2, sid=sid)   # ← 「同情/助言」ごとの短文ペア


            # SYSTEMは初回に焼いてあるので、今回は user だけ送る
            payload = style_hint + "\n\n" + user_msg
            r = self.client.responses.create(
                model=self.model,
                conversation=conv_id,
                input=[*shots, {"role": "user", "content": payload}],
                temperature=0.65,
                top_p=0.9, 
                max_output_tokens=REPLY_TOKENS,
            )
            text = (r.output_text or "").strip()
        except Exception as e:
            text = f"ごめん、今ちょい不調…！（{e}）"


        return _bubble_split(text, max_bubbles=max_bubbles)


# プロフィール入力ページ
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'POST':
        # フォームからプロフィール情報を取得
        nickname = request.form.get('nickname')
        hobby = request.form.get('hobby')
        job = request.form.get('job')
        personality = request.form.get('personality')
        # セッションに保存
        session['profile'] = {
            'nickname': nickname,
            'hobby': hobby,
            'job': job,
            'personality': personality
        }
        # チャット画面へリダイレクト
        return redirect(url_for('chat'))
    return render_template('profile.html')

# チャット画面のルート
@app.route('/chat')
def chat():
    user_profile = session.get('profile')
    # プロフィール情報を初期プロンプトに反映する処理をここに追加予定
    return render_template('gal_index.html', profile=profile)

# トップページ（必要ならリダイレクト）
@app.route("/")
def index():
    return redirect(url_for('profile'))

@app.route("/ask", methods=["POST"])
def ask():
    payload = request.json or {}
    # セッションIDはクッキー→無ければUUIDで
    sid = request.cookies.get("sid") or payload.get("sid") or str(uuid.uuid4())
    sess = get_session(sid)
    import logging
    user_profile = session.get('profile')
    logging.warning("DEBUG: /ask profile = %s", user_profile)

    user_msg = (payload.get("message") or "").strip()
    if not user_msg:
        return jsonify({"sid": sid, "answer": "え、なんて？💦"})

    # 1) まず user を履歴に積む（ここが最優先）
    sess["history"].append({"role": "user", "content": user_msg})
    sess["last_user_text"] = user_msg  # （任意）直近ユーザー保持

    # 2) 直前入力だけ見て返答を生成（プロフィール情報を渡す）
    responder = Response(client, model="gpt-4.1", system_prompt=SYSTEM_PROMPT, profile=user_profile)
    intent = responder.classify_intent(user_msg)
    target = responder.plan_target(sess["gmd_totals"][-1] if sess.get("gmd_totals") else None)
    bubbles = responder.generate_reply(
        user_response=user_msg,
        intent=intent,
        target_score=target,
        focus_dimensions_display=[],
        max_bubbles=2,  # ← 3 → 2 にして冗長化を抑制
        profile=user_profile
    )

    # 3) assistant は先頭バブルだけ保存（1メッセージでOK）
    if bubbles:
        sess["history"].append({"role": "assistant", "content": bubbles[0]})
        sess["last_ai_text"] = bubbles[0]


    # 返却ペイロード（まず返答）
    resp_payload = {"sid": sid, "answer": bubbles}

    # 4) ★50字“発言終端”採点：前回採点以降のユーザ発言だけでカウント
    ctx, new_last_u = find_scoring_span_user_only(
        history=sess["history"],
        last_user_idx=sess.get("last_scored_user_idx", -1),
        threshold=50,
    )

    if ctx is not None:
        scorer = Scoring(client, model="gpt-4.1", window_chars=50)
        result = scorer.score_from_context(ctx)

        # セッション更新
        sess["last_scored_user_idx"] = new_last_u
        sess["gmd_totals"].append(result["total"])
        sess["gmd_details"].append(result["details_display"])
        sess["eval_count"] += 1

        # クライアント向け（新形式 gmd オブジェクト）
        resp_payload["gmd"] = {
            "total": result["total"],
            "details": result["details_display"],
            "eval_index": sess["eval_count"] - 1,
            "context_chars": result["window_chars"],
            "context_excerpt": result["context_excerpt"],
        }

    response = jsonify(resp_payload)
    response.set_cookie("sid", sid, max_age=60*60*24*30, httponly=True, samesite="Lax")
    return response, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)
