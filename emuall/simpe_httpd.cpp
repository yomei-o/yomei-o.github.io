#include <iostream>
#include <filesystem>
#include "./httplib.h" // 同じディレクトリに httplib.h を置いて用意します

namespace fs = std::filesystem;

int main() {
    // カレントディレクトリ（絶対パス）を取得
    std::string current_dir = fs::current_path().string();
    
    // HTTPサーバーのインスタンスを作成
    httplib::Server svr;

    // 静的ファイルの配信ディレクトリを設定
    // 第二引数にルートパス（"/"）へのアクセス時に自動で返すインデックスファイル（index.html）を指定
    if (!svr.set_mount_point("/", current_dir, "index.html")) {
        std::cerr << "エラー: ディレクトリのマウントに失敗しました。" << std::endl;
        return 1;
    }

    std::cout << "サーバーを起動しました: http://localhost:8000" << std::endl;
    std::cout << "配信ディレクトリ: " << current_dir << std::endl;

    // 0.0.0.0 のポート8000でサーバーを待機状態にする
    if (!svr.listen("0.0.0.0", 8000)) {
        std::cerr << "エラー: ポート8000でのリスンに失敗しました。" << std::endl;
        return 1;
    }

    return 0;
}