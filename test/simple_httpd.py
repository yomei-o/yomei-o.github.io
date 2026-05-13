import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# カレントディレクトリの絶対パスを取得
current_dir = os.getcwd()

# StaticFilesをマウントする
# directory: 公開するフォルダのパス
# html=True: インデックスファイル（index.html）を自動で探す設定
app.mount("/", StaticFiles(directory=current_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # サーバーを起動（ポート8000）
    uvicorn.run(app, host="0.0.0.0", port=8000)