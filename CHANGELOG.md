# 変更履歴

## 0.2.3 — 2026-10-25

- 新規初期値をRank / Alpha 16 / 16、sigma・学習用flow shift 1.0、2500STEPへ変更。100STEP保存・保存済み設定は維持。

- Weight decayとKohaku専用EMA decayを追加。EMA有効時は通常版とは別の平均化重みを出力。「キャラクターLoKr」プリセットで記事の対応設定を一括適用。機能・設定値が実験的であることをUI・ガイドへ明記。

- Kohaku / sd-scriptsの学習方式にLoRA・LoKr・LoHa選択とLoKr factorを追加。既存設定はLoRAを維持。
- 両エンジンにpolynomial Schedulerとpowerを追加。既定のcosineを維持。
- 方式・factor・powerのstate再開チェック、設定の保存・プレビュー・出力、日本語・英語のガイドを更新。

## 0.2.2 — 2026-10-24

- かんたん導入にsd-scripts用xFormers / Flash Attentionの任意導入を追加。初期状態は両方OFF。対応xFormersの自動選択、Flash Attentionのローカルwheel指定、導入後のGPU順伝播・逆伝播検証に対応。Attentionの選択値は維持。

- KohakuVault 1.0でキャッシュ読み込みが `read_range` エラーになる問題を修正。公開APIによる一括読み込みへ対応し、従来版も維持。

- 起動時に最後に開いたプロジェクトを復元。名称変更後もIDで特定し、対象がない場合はデフォルト設定の新規プロジェクトを作成します。

- UIを琥珀色の配色へ変更。クリーム色の背景、焦げ茶のヘッダーとログ、琥珀色のボタン・タブ・アイコンへ統一。

- 名称をMOSS LoRA Trainer with Kohakuへ統一。UI・起動時エラー・ログ・ドキュメント・配布ZIPを更新し、起動ファイルをStart MOSS LoRA Trainer.batへ変更。

- 公開後の調整: 衣装の学習結果を踏まえ、デフォルト学習率を1e-4へ戻しました。その他の初期値は維持。

- cosine_with_restartsを両エンジンへ追加。初期値2周期、warmupは開始時のみ。通常のcosineを既定に維持し、周期数の入力・出力・state再開の差分チェックに対応。

- CAMEを両エンジンへ追加し、導入処理・依存チェック・Kohaku連携を追加。新規Optimizerは両エンジンともAdamW、Schedulerはcosine・warmup 100。

- Kohakuでshift timestepに対応。新規初期値をshift・flow shift 3.0へ変更し、sd-scriptsの設定ガイドにも反映。既存のsigma設定とstate互換性チェックを維持。

- Caption dropoutに自然文キャプションでは0推奨と表示。STEP数0の入力で編集欄が無効になる問題を修正し、STEP指定時は未使用のEpoch数0を許可。

- Kohakuの学習対象にFFN_downのみ・Attention＋FFN・Attentionのみの選択を追加。新規初期値をFFN_downのみ・学習率1e-4・2000STEP・100STEPごと保存に変更。
- 旧プロジェクトの学習対象をAttention＋FFNとして保持し、state再開時の対象変更を検出。sd-scriptsの包含・除外設定と速度調整を設定ガイドへ追記。
- 両エンジンの新規STEP数を2000・保存間隔を100に統一。日英ドキュメントと設定ガイドを整理。
- 利用者より、RTX 4070 12GBで初回公開時のデフォルト設定の学習完走を報告。

## 0.2.1 — 2026-10-21

- 利用者のRTX 4070 12GBで、Kohakuの4000STEP学習完走・サンプル生成・最終LoRA出力を確認。READMEと検証記録に設定条件・実測値を追記。

- Kohakuのサンプル画像で幅・高さが逆転する問題を修正。縦長・横長の設定を正しい高さ・幅の順で出力。

- Alphaの整数値は末尾の.0を省略して表示。小数値の入力・保存は維持。

- 確認操作を画像・キャプション確認、学習前チェック、実行設定プレビューに整理。エクスポートは実行・ログへ移動。
- 設定ガイドを設定値・使い方・モデル指定に絞って簡潔化。

- Kohaku（旧設定・新設定）とsd-scriptsの両バックエンドで1500STEPの学習完走を、利用者のRTX 4070 12GB実機で確認した旨を記載。

- 新規学習条件をKohaku公式LoRAレシピを参考に、32/32・AdamW8bit・cosine・warmup100・下限10%・4000step・200step保存（Optimizer・保存間隔はGUI独自値）・caption dropout0.1へ変更。
- cosine下限をGUIで設定可能にし、既存プロジェクトの下限0%とstate再開互換性を保持。
- 設定ガイド・README・日英UIを更新し、sd-scriptsの開始値と過去の速度測定条件を明記。

## 0.2.0 — 2026-10-20

- 主要表示名をMOSS LoRA Trainer with Kohakuに変更。KohakuMossのライセンス本文・通知を収録。
- Kohaku専用環境、Tritonの自動導入とGPU診断、キャッシュ、学習、checkpoint再開に対応。
- 新規プロジェクトのエンジンをKohaku、メモリ構成をckpt-residentへ変更。既存設定は保持。
- AdamW8bitはGPU状態・非fused更新、自動チューニングOFF。子プロセスはUTF-8で起動。
- 確認済みGPU AdamW8bit checkpointでは、メモリ構成だけ変更して再開可能。
- 保存前にLoRA exportを退避し、state整理後もモデル履歴を保持。
- 起動ファイルをStart MOSS LoRA Trainer.batへ統一。不要な旧バッチと重複メモを整理。
- アプリ・主要操作・タブのアイコンを刷新。
- 設定表は独立したMarkdownへ移し、GUIには設定ガイドボタンを配置。
- モデル案内を設定ガイドへ、Kohaku説明を利用ガイドへ、検証記録を開発文書へ統合。

## 0.1.x

- sd-scriptsによるMOSS-Image LoRA学習、独立したセットアップ環境、VRAMプリセット。
- プロジェクト作成・コピー・削除、設定保存・書き出し・読み込み。
- state再開と追加学習、進捗・残り時間・VRAM表示、通知音。
- 日本語・English切り替え、キャプション検査、CPUスレッド数設定。
- 小さい画面での下部操作固定、スクロール操作と選択欄の挙動改善。
