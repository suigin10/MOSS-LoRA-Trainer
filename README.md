# MOSS Studio with Kohaku v0.2.2

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

新規Kohakuは、**FFN_downのみ、Rank/Alpha 32/32、AdamW、学習率5e-5、cosine、warmup 100、最小学習率10%、2000STEP、100STEPごと保存、timestep shift / flow shift 3.0**が初期値です。Caption dropoutは0.1で、自然文キャプションでは0を推奨します。
学習対象は「03 学習」でattn-mlp（Attention＋FFN）・attn（Attentionのみ）にも切り替えられます。OptimizerにはAdamW8bit・CAMEも選べます。CAME用パッケージは「かんたん導入」で追加します。
INT8・ckpt-resident・batch 1・画像＋テキスト上限6144は、このGUIの12GB向け開始設定です。
Schedulerは通常のcosineが既定です。比較用にcosine_with_restartsも選択でき、初期値は2周期（再スタート1回）です。詳細は設定ガイドを参照してください。

両エンジンとも、新規設定のSTEP数は2000、保存間隔は100、OptimizerはAdamW、Schedulerはcosineです。sd-scripts用のRank・学習率などは [設定ガイド](docs/STABLE_SETTINGS.md) の別列を参照してください。
既存プロジェクトは保存済みの値を保持します。更新やエンジン切り替えだけでは学習条件を上書きしません。学習対象・Optimizer・Scheduler・timestep方式を変える場合は、新規学習を開始してください。

## 設定と出力

- プロジェクト: `data/projects/<ID>/project.json`
- 実行環境: `data/environment.json`
- 実行ログ・生成設定: `data/projects/<ID>/runs/<実行ID>/`
- LoRA出力: 実行フォルダの `models/`。KohakuはComfyUI形式とLyCORIS形式を保存します。
- Kohaku再開state: 実行フォルダの `kohaku/checkpoints/step_XXXXXXXX/`

Kohakuとsd-scriptsのstateは相互利用できません。更新時はアプリを終了し、`data/`・`backend/`・モデルを保持してください。`setup_backend.py` は自動導入に必要です。

## 現在の確認状況

- 利用者のRTX 4070 12GBで、現行デフォルト設定の学習完走を報告いただいています（2026-10-24）。
- 過去にはKohakuの4000STEP学習・サンプル生成・最終LoRA出力、両エンジンの1500STEP学習も完走しています。過去の速度値は現行設定と条件が異なります。
- CAMEと再スタート付きcosineはCPUで更新・state復元を確認しています。GPUでの生成品質比較は未実施です。

自動テスト、実機報告、過去の速度値と条件は [開発・検証](CONTRIBUTING.md#検証状況) にまとめています。

## ライセンス・クレジット


GUIのソースと同梱通知音は [MIT License](LICENSE) です。学習エンジンとモデルは、それぞれの配布元のライセンスに従います。Kohakuは名称条件付きのKohakuMoss License 1.0です。主要表示名「MOSS Studio with Kohaku」はこの条件に対応しています。[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) も参照してください。

- [Anima Standalone Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer)：ローカル学習GUI・プロジェクト管理の参考
- [sd-scripts MOSS-Image学習ガイド](https://github.com/kohya-ss/sd-scripts/blob/main/docs/moss_image_train_network.md)：引数・モデル形式・VRAM設定の参考
- [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)：対象モデル

本プロジェクトは独立したGUIであり、上記プロジェクトの公式製品ではありません。学習エンジンのソースとモデル本体は配布ZIPに同梱せず、エンジンはセットアップ時に取得します。Kohakuのライセンス本文は `licenses/` に収録しています。

