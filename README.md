# MOSS Studio — MOSS-Image LoRA Trainer

**MOSS-ImageのLoRA学習を、プロジェクト作成から実行・再開まで操作できるWindows向け日本語GUIです。** 学習エンジンには [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) を使用します。

A Japanese desktop GUI for MOSS-Image LoRA training on Windows. Includes isolated backend setup, project management, VRAM presets, resume support, and an estimated completion time.

> **初期プレビュー版**：RTX 4070（VRAM 12GB）で実機検証済みです（提供者による確認）。GUI・設定・導入制御などの自動テストも実施しています。検証範囲は [VALIDATION.md](VALIDATION.md) を参照してください。

## ダウンロード・導入

1. [Releases](https://github.com/suigin10/MOSS-LoRA-Trainer/releases) から `MOSS-LoRA-Trainer.zip` をダウンロードして展開します。
2. **Python 3.12（64bit）**を用意し、`install.bat` を起動します。
3. 「導入・確認・修復を開始」を押します。初回は専用venvを作成し、sd-scriptsとCUDA版PyTorchなどを導入します。
4. モデル・学習画像を指定して、`start.bat` から利用します。

GitHubの「Code → Download ZIP」から取得したソースでも同じ手順で起動できます。Gitのインストールは不要です。2回目以降の導入操作は、同じ専用venvを確認・修復します。

## 動作環境

| 項目 | 対象 |
|---|---|
| OS | Windows 10 / 11、64bitを想定 |
| Python | GUIは3.11以上。自動導入用は64bitの3.11 / 3.12（3.12推奨） |
| GPU | CUDA・bf16対応のNVIDIA GPU |
| VRAM | 12GBを開始構成とし、16GB / 24GB以上のプリセットを用意 |
| モデル | MOSS-Image DiT、Qwen3.5-9B、FLUX.2 VAEを別途取得 |
| その他 | 初回導入時のネット接続、依存パッケージ・モデル用のディスク空き容量 |

VRAMだけで動作を保証するものではありません。画像解像度・キャプション長・CPUスワップ・RAM容量によって必要なメモリは変わります。8GB向けプリセットは用意していません。

## 主な機能

- モデルと画像をWindowsのファイル／フォルダ選択ダイアログから指定
- 初期値からの新規プロジェクト作成、既存プロジェクトの設定コピー
- STEP数／エポック数の切り替え、設定の保存・書き出し・読み込み
- VRAM容量別のプリセット、設定の競合チェック
- LoRA重みからの追加学習、保存したstateからの再開、元設定との比較
- サンプル生成パラメーター、ログ・進捗・残り時間・推定終了時刻
- 完了／失敗時の「ポ・コ・ピン♪」通知音とON／OFF・試聴

## 初期設定

| 項目 | 初期値 |
|---|---|
| 学習上限 | 1,500 STEP（直接変更可能） |
| 学習率 / Rank / Alpha | 1e-4 / 16 / 1 |
| Optimizer / scheduler | AdamW8bit / constant |
| 保存 | 100 STEPごと、再開用stateも保存 |
| 解像度 / batch | 1024 / 1 |
| 12GB向け設定 | DiT・QwenともConvRot INT8、block swap 20、ディスクキャッシュ |
| サンプル生成 | 初期OFF。1024×1024、50 steps、CFG 3、生成shift 5 |

初期値は調整の開始点です。学習対象・データ量に応じて変更してください。

## ドキュメント

- [利用ガイド](docs/USER_GUIDE.ja.md)：導入・設定・レジューム・ログ・通知音
- [VRAM別の推奨モデル](推奨モデル.txt)：具体的なファイル名・ダウンロード先
- [検証範囲](VALIDATION.md)：実施済みテストと未検証の項目
- [開発・不具合報告](CONTRIBUTING.md)
- [変更履歴](CHANGELOG.md)

## 設定の保存場所と更新

プロジェクト・実行結果は `data/`、自動導入した環境は `backend/moss-env/` に保存します。アプリを更新するときは終了してから配布ZIPを同じフォルダへ上書き展開してください。`data/` と `backend/` は配布ZIPに含みません。

設定の共有は「設定を書き出す」で行います。読み込みには書き出し先の `project_snapshot.json` を選びます。別PCではモデル・画像のパスを選び直してください。モデル本体・学習データ・実行結果はリポジトリに含みません。

## ライセンス・クレジット

GUIのソースと同梱通知音は [MIT License](LICENSE) です。学習エンジンとモデルは、それぞれの配布元のライセンスに従います。[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) も参照してください。

- [Anima Standalone Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer)：ローカル学習GUI・プロジェクト管理の参考
- [sd-scripts MOSS-Image学習ガイド](https://github.com/kohya-ss/sd-scripts/blob/main/docs/moss_image_train_network.md)：引数・モデル形式・VRAM設定の参考
- [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)：対象モデル

本プロジェクトは独立したGUIであり、上記プロジェクトの公式製品ではありません。sd-scriptsのソースはリポジトリに同梱せず、セットアップ時に取得します。
