# 開発と不具合報告

## テスト

Python 3.12とtkinterを使用します。GUI・単体テストの実行に学習用モデルやPyTorchのインストールは不要です。

```powershell
python -m unittest discover -s tests -v
```

GUIテストはウィンドウを生成し、通知音APIをモック化します。Windowsでの実行を想定しています。`MOSS_TEST_TMP` 環境変数でテスト用一時フォルダの親ディレクトリを指定できます。

## 不具合報告に含める情報

- 再現手順と期待した動作・実際の動作
- アプリの版、Windows・Python・GPU・VRAM
- 学習用PyTorch・CUDA・sd-scriptsの版（分かる範囲）
- 使用モデルのファイル名、関係する設定、ログの該当部分

実行ログは `data/projects/<ID>/runs/<実行ID>/training.log`、導入ログは `data/setup-logs/` にあります。共有前に個人名・ローカルパス・トークンなど不要な情報を除いてください。設定共有には書き出した `project_snapshot.json` を使えますが、モデル・画像のパスも含まれる点に注意してください。

GPU学習の完走結果は歓迎します。成功条件・モデル・設定とともに報告し、GUIの単体テストと実機の動作確認は区別して記録してください。
