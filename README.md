# ギャルマイン度リアルタイム評価チャット

このプロジェクトは、ユーザーの発言からギャル風AIがチャット応答し、対話の内容をもとに「ギャルマイン度」を評価するWebアプリです。

## 🔥 システム概要

- FlaskとClaude API（sonnet-4.6）を使った日本語チャットシステム
- ユーザーの入力に対してギャル口調の返答を生成
- 会話ごとに心理的指標を評価し、ギャルマイン度（0〜50点）を算出
- 8つの心理特徴に基づく内部評価を表示
- ブラウザ上で会話とスコアを確認可能

## 💡 特徴

- Claude sonnet-4.6モデルを使った自然な日本語応答
- 口調や雰囲気を「ギャル風」に寄せたシステムプロンプト
- 会話の流れを保持しつつ、スコアリング用の文脈を抽出
- 5つの対話ごとにスコアを更新して推移を可視化

## 🧩 使用技術

- Python 3.10+ / 3.11+
- Flask
- Flask-CORS
- python-dotenv
- numpy
- gunicorn（Herokuや公開用デプロイ時）

## 🔧 必要な準備

1. リポジトリをローカルにクローン
2. Python仮想環境を作成し、有効化
3. 依存パッケージをインストール
4. Claude APIキーを環境変数に設定

### 例: Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Claude APIキーの設定

プロジェクトルートに `.env` ファイルを作成し、次の内容を追加します。

```dotenv
CLAUDE_API_KEY=your_claude_api_key
```

または、直接環境変数として設定します。

## ▶️ 起動方法

### 開発環境での実行

```powershell
python app.py
```

起動後、通常は次のURLにアクセスします。

```
http://127.0.0.1:5000
```

### Gunicornを使った実行（本番向け）

```powershell
gunicorn app:app
```

## 📝 使い方

1. ブラウザで `/profile` にアクセス
2. ニックネームを入力してスタート
3. チャット画面にメッセージを送信
4. 5回の会話ごとにギャルマイン度が更新される

## ⚠️ 注意点

- Claude APIキーが必要です
- 無料枠を超えるリクエストでは課金が発生します
- 現在のモデル設定は `sonnet-4.6` です

## 📁 主要ファイル

- `app.py` - Flaskアプリ本体
- `requirements.txt` - Python依存パッケージ
- `templates/gal_index.html` - チャット画面テンプレート
- `templates/profile.html` - プロフィール入力画面
- `static/script.js` / `static/style.css` - フロントエンド実装
- `Procfile` - Gunicorn起動設定

## 📌 補足

- `Procfile` には `web: gunicorn app:app` が指定されています
- そのままHerokuや類似PaaSへデプロイ可能です

---

このREADMEは、システムの起動方法と利用方法を最新版の実装に合わせて更新しています。