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

## 検証状況

確認日: 2026-10-21。Windows / Python 3.12 / RTX 4070 12GB。

- resident初期値の変更後、自動テスト146件成功。設定ガイドをMarkdownへ移した後、GUI・翻訳8件とガイドボタンの読み込みを再確認。
- 利用者のRTX 4070 12GB実機で、旧設定の1500STEP学習完走を確認（2026-10-21報告）。100STEP学習と50STEP追加学習も成功。
- INT8・AdamW8bit・auto精度の各12STEP診断で、従来構成4.518秒/step、ckpt-resident4.102秒/step。residentのデバイスVRAMは約10.3GiB。lossは有限。
- 実際の150STEP checkpointからresidentで153STEPまで再開成功。
- FP8比較は3.370秒/stepだったが生成品質は未比較。標準設定には採用しない。
- 新しい公式準拠設定（32/32・AdamW・cosine）での長時間完走、新規専用venvへのネット経由導入完走、最終的な生成品質比較は未確認。

自動テストとGPU実機診断は別の確認です。導入・失敗時の登録防止などにはモックを使用しています。
各値は同じ実データでの短期測定であり、他の画像やキャプション・GPUの速度を保証しません。

### 参照ソース

- MOSS-LoRA-Trainer: 6a12d8bbce7340a990db7152abfb39e78a5d2d7c
- KohakuMossTrainer固定先: 0c4737e3b8787a7df0f2d28a2972640a5f8522f6

### torch.compileの診断

同じINT8・ckpt-resident・実データの20STEP比較では、OFFが20STEP完走（2〜20STEP中央値4.091秒）。
ONは既定設定とLRU並べ替え無効化で最初のSTEPにCheckpointError（forward再計算の保存テンソル数67対62）が発生。
LRU無効＋再コンパイル上限64では1STEPが138.3秒（compile111.4秒）で通ったが、2STEP目で上限へ到達後に同じエラー。
ONの安定速度は未測定であり、初期値はOFFを維持する。回避策は診断限定でアプリには適用しない。

### 2026-10-21 公式初期値への整合

新規初期値とcosine下限、既存プロジェクトの下限0%移行、state再開差分チェック、sd-scripts書き出し、日本語・英語UIを含む148テスト成功。32/32・AdamW・cosineのGPU速度・VRAM・生成品質比較は未実施。
