# 変更履歴

## 0.2.0 — 2026-10-20

- 主要表示名をMOSS Studio with Kohakuに変更。KohakuMossのライセンス本文・通知を収録。
- Kohaku専用環境、Tritonの自動導入とGPU診断、キャッシュ、学習、checkpoint再開に対応。
- 新規プロジェクトのエンジンをKohaku、メモリ構成をckpt-residentへ変更。既存設定は保持。
- AdamW8bitはGPU状態・非fused更新、自動チューニングOFF。子プロセスはUTF-8で起動。
- 確認済みGPU AdamW8bit checkpointでは、メモリ構成だけ変更して再開可能。
- 保存前にLoRA exportを退避し、state整理後もモデル履歴を保持。
- 起動ファイルをStart MOSS Studio.batへ統一。不要な旧バッチと重複メモを整理。
- アプリ・主要操作・タブのアイコンを刷新。
- 設定表は独立したMarkdownへ移し、GUIには設定ガイドボタンを配置。
- モデル案内を設定ガイドへ、Kohaku説明を利用ガイドへ、検証記録を開発文書へ統合。

## 0.1.x

- sd-scriptsによるMOSS-Image LoRA学習、独立したセットアップ環境、VRAMプリセット。
- プロジェクト作成・コピー・削除、設定保存・書き出し・読み込み。
- state再開と追加学習、進捗・残り時間・VRAM表示、通知音。
- 日本語・English切り替え、キャプション検査、CPUスレッド数設定。
- 小さい画面での下部操作固定、スクロール操作と選択欄の挙動改善。
