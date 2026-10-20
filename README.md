# MOSS Studio with Kohaku v0.2.1

Windows向けのMOSS-Image LoRA学習GUIです。Kohaku / sd-scriptsの専用環境導入、プロジェクト管理、キャッシュ、学習・追加学習、ログ表示に対応します。新規プロジェクトはKohakuを使用します。既存プロジェクトのエンジンと保存設定は維持します。

Japanese / English desktop GUI for MOSS-Image LoRA training. Switch to English in the header. See the [English quick start](docs/QUICK_START.en.md).

## 起動と導入

1. Python 3.12（64bit）を用意し、`Start MOSS Studio.bat` を起動します。
2. 「かんたん導入」で学習エンジンと専用環境を作成します。
3. モデルと画像を指定し、設定を確認して学習を開始します。

CUDA / bf16対応のNVIDIA GPUが対象です。Kohakuの導入には対応ドライバとVisual Studio Build ToolsのC++ワークロードが必要です。GUI自体はPython標準のtkinterを使い、学習用パッケージは専用venvへ導入します。モデルは別途用意してください。

## ドキュメント

- [設定ガイド](docs/STABLE_SETTINGS.md): 12GB向けKohaku / sd-scripts設定表、VRAMプリセット、モデルのファイル名とダウンロード先。
- [利用ガイド](docs/USER_GUIDE.ja.md): 導入、プロジェクト、キャプション、学習・再開、出力形式、Kohaku固有の対応範囲。
- [English quick start](docs/QUICK_START.en.md)
- [開発・検証](CONTRIBUTING.md): テスト方法、不具合報告、実機検証結果と未確認項目。
- [変更履歴](CHANGELOG.md)

設定表は「04 VRAM・速度」の「設定ガイド」ボタンからも開けます。
「学習前チェック」はパスと設定の不整合を検査します。画像・キャプションの詳細はデータセットタブ、実行設定のプレビューと設定エクスポートは「実行・ログ」にあります。

## 新規学習の初期値

Kohakuの学習条件は公式LoRAレシピを参考に、**Rank/Alpha 32/32、AdamW、LR 1e-4、cosine、warmup 100、最小学習率10%、1500step、100stepごと保存、caption dropout 0.1**を使用します。
学習上限1500step・保存間隔100stepは、使いやすさを優先したこのGUI独自の初期値です（公式共通設定は4000step・500step保存）。
INT8・ckpt-resident・batch 1・画像＋テキスト上限6144は、このGUIの12GB向け開始設定です。

既存プロジェクトとstateの保存設定は維持します。旧cosineの最小学習率は従来の0%を保持します。
sd-scriptsの推奨開始値は [設定ガイド](docs/STABLE_SETTINGS.md) の別列を参照してください。エンジン切り替えだけで学習条件は上書きしません。

## 設定と出力

- プロジェクト: `data/projects/<ID>/project.json`
- 実行環境: `data/environment.json`
- 実行ログ・生成設定: `data/projects/<ID>/runs/<実行ID>/`
- LoRA出力: 実行フォルダの `models/`。KohakuはComfyUI形式とLyCORIS形式を保存します。
- Kohaku再開state: 実行フォルダの `kohaku/checkpoints/step_XXXXXXXX/`

Kohakuとsd-scriptsのstateは相互利用できません。更新時はアプリを終了し、`data/`・`backend/`・モデルを保持してください。`setup_backend.py` は自動導入に必要です。

## 現在の確認状況

**旧設定で1500STEPの学習完走を、利用者のRTX 4070 12GB実機で確認済みです（2026-10-21報告）。** 100STEP学習と50STEP追加学習も確認済みです。旧設定（Rank16/Alpha1・AdamW8bit・constant、INT8・ckpt-resident）の短期診断は約4.10秒/step、デバイスVRAM約10.3GiBでした。**新設定（Rank32/Alpha32・AdamW・cosine）でも300STEPの動作確認済みです（利用者報告、2026-10-21）。** 旧設定の速度・VRAM測定値は、新設定の測定値とは区別しています。生成品質の比較と、新しい公式準拠設定での長時間完走は未確認です。詳細は開発・検証文書にまとめています。

## ライセンス・クレジット


GUIのソースと同梱通知音は [MIT License](LICENSE) です。学習エンジンとモデルは、それぞれの配布元のライセンスに従います。Kohakuは名称条件付きのKohakuMoss License 1.0です。主要表示名「MOSS Studio with Kohaku」はこの条件に対応しています。[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) も参照してください。

- [Anima Standalone Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer)：ローカル学習GUI・プロジェクト管理の参考
- [sd-scripts MOSS-Image学習ガイド](https://github.com/kohya-ss/sd-scripts/blob/main/docs/moss_image_train_network.md)：引数・モデル形式・VRAM設定の参考
- [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)：対象モデル

本プロジェクトは独立したGUIであり、上記プロジェクトの公式製品ではありません。学習エンジンのソースとモデル本体は配布ZIPに同梱せず、エンジンはセットアップ時に取得します。Kohakuのライセンス本文は `licenses/` に収録しています。

