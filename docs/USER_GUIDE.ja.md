# MOSS Studio with Kohaku 利用ガイド

MOSS-Image 専用の日本語デスクトップ GUI です。Windows の Python 3.11 以上で `Start MOSS Studio.bat` をダブルクリックして起動します。GUI 自体に追加の pip パッケージは不要です（tkinter が必要）。学習環境は「かんたん導入」で作成できます。モデルファイルは別途用意してください。

対象GPUは CUDA / bf16 対応の NVIDIA ビデオカードです。推奨設定は特定の機種名ではなく、VRAM容量別に用意しています。新規プロジェクトのエンジンはKohakuです。初期値は VRAM 12GB のビデオカード向けで、16GB・24GB以上のプリセットも選べます。

画面はネイビーとティールを基調にしています。白い欄は直接入力、淡いティールの欄はクリックして選択、グレーの欄は無効状態です。選択中のタブや主要操作は濃いティールで表示します。

設定ページ内のホイール操作はページのスクロールに使います。閉じたOptimizer等のドロップダウン上でも値は変わりません。値を変える場合はリストを開いて選択してください。

**モデル選びは同梱の [設定・モデルガイド](STABLE_SETTINGS.md) を参照してください。** VRAM 12GB・16GB・24GB以上ごとの具体的なファイル名、ダウンロード先、GUIプリセット、bf16分割ファイルの指定方法を記載しています。

## かんたん導入（推奨）

**Python 3.12（64bit）をインストール済みなら、`Start MOSS Studio.bat` を開き、GUI右上の「かんたん導入」からエンジンを選び「導入・確認・修復を開始」を押します。** GUI 右上の「かんたん導入」からも同じ操作ができます。

- 初回だけ専用 venv と sd-scripts を作成します。2 回目以降は、この GUI が作った同じ専用環境を確認・修復します。他用途の venv やグローバルパッケージは流用しません。
- sd-scripts は公式リポジトリの ZIP から取得するため Git は不要です。
- CUDA 版 PyTorch、sd-scripts の requirements、TensorBoard、bitsandbytes を順に導入します。
- CUDA / bf16 と MOSS スクリプトの起動をチェックし、成功した場合だけ GUI の Python / sd-scripts 設定を登録します。
- 途中失敗・停止後も同じ導入先で再試行できます。フォルダは増やさず、既存の学習結果を削除しません。専用環境かどうかを識別ファイルで確認し、不明な環境は変更せず停止します。

標準は PyTorch 2.8.0 / CUDA 12.8（`cu128`）です。ドライバとの組み合わせに応じて PyTorch 2.6.0 / CUDA 12.4（`cu124`）も選べます。NVIDIA ドライバや Python 自体のインストールは自動では行いません。3.12 / 3.11 の64bit Pythonを自動検出できない場合は、「作成用 Python」の参照ボタンで指定します。

取得にはインターネット接続と数GB以上の空き容量が必要です。通常の導入先は `backend/moss-env/`、venv はその中の `venv/` です。再実行時にも同じ導入先・CUDA プロファイルを初期表示します。取得済み sd-scripts は版を維持し、依存パッケージを確認・修復します。修復時には専用 venv 内のパッケージが変更されることがあります。失敗時に GUI の参照パスは変更しませんが、修復中のパッケージ変更をロールバックする方式ではありません。

ログは `data/setup-logs/`、取得した sd-scripts の commit と環境情報は `installation.json`、パッケージ一覧は `packages.txt` に残します。モデル本体はダウンロードしません。開発版0.2ではTritonをPyTorchの版に合わせて自動導入し、GPUカーネルのコンパイル・実行を確認します。Kohakuの導入と対応範囲は [Kohakuバックエンド](#kohakuバックエンド) を参照してください。

## 最初に使うとき

1. ZIP を展開し、`Start MOSS Studio.bat` を起動します。コマンドからは `python launch.py`。
2. 「かんたん導入」で新規環境を作ります。手動で準備した環境を使う場合は、右上の「実行環境」で Python と sd-scripts フォルダを指定できます。
3. 「保存して環境チェック」で CUDA / bf16 / パッケージを確認します。
4. 「モデル」タブの各「参照…」で、DiT、Qwen3.5-9B、FLUX.2 VAE をファイル選択ダイアログから指定します。手入力もできます。
5. 「データセット」で画像フォルダを選びます。同じフォルダに `image.png` と `image.txt` を置き、テキストはUTF-8で保存してください。複数フォルダはサブセットを追加します。
6. 「VRAM・速度」タブで必要に応じてプリセットを選択し、学習対象に合わせて STEP 数などを調整します。「設定を確認」後に「学習を開始」。

### 手動で学習用環境を準備する場合

既に MOSS 対応 sd-scripts の環境があればそれを指定できます。Anima の trainer フォルダそのものを指定する構成ではありません。

新しく用意する場合の流れ（PowerShell、GUI のフォルダで実行）:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git backend/sd-scripts
py -3.12 -m venv backend/venv
```

次に、[PyTorch 公式のインストール案内](https://pytorch.org/get-started/locally/)に従って、**`backend/venv/Scripts/python.exe -m pip` を使い** GPU・ドライバに対応する CUDA 版 PyTorch をインストールします。その後:

```powershell
backend/venv/Scripts/python.exe -m pip install -r backend/sd-scripts/requirements.txt
backend/venv/Scripts/python.exe -m pip install tensorboard bitsandbytes
```

sd-scripts の環境要件を優先してください。GUI と学習用 Python は別の環境でも構いません。INT8 の高速カーネルには、PyTorch と対応する Triton 環境が必要です。学習開始時に不足パッケージや起動エラーがあればログに表示します。環境チェックは全ライブラリの ABI 互換性やモデルの読込成功までは保証しません。

## Captionの確認

「画像数・Caption数を確認」で「Caption 認識 N / M 件」を表示します。Mは画像数、Nは対応ファイルをUTF-8で読み取れて本文もあるCaptionの数です。「caption なし 0 枚」は欠落が0枚という意味です。空の本文やUTF-8以外のテキストは、ファイルが見つからない場合とは別に表示します。

例は `image.png` と同じ場所の `image.txt` です。`image.png.txt` や `image.txt.txt` は読み取り対象になりません。見つからない場合は、期待する名前を確認画面で表示します。画像が入っているフォルダ自体を選択してください。親フォルダからの再帰走査は行いません。`10_subject` のようなフォルダ名でも、repeat数は画面の「Repeat」で明示的に設定します。確認処理は画像・Captionを変更しません。

## 初期値と変更方法

新規学習はKohaku公式の32/32・AdamW・cosineを使用します。全設定とsd-scriptsの開始値は [設定ガイド](STABLE_SETTINGS.md) を参照してください。
UIの値は保存済みプロジェクトの設定です。更新だけで既存値を公式初期値へ上書きしません。
キャプション単体の上限は「01モデル」、画像＋テキストの合計上限は「04 VRAM」のKohaku設定です。

### LoRAとstateの保存

LoRAの保存間隔は自由に変更できます。例: 50 STEPごと、250 STEPごと、1エポックごと。STEP間隔とエポック間隔を両方指定することもできます。0にした間隔は無効です。過去のLoRAファイルはstate保持設定では削除しません。

再開用stateはモデル保存時に作成し、初期設定の「state は最新の1つだけ保持」がONなら、この実行中に正常保存した最新1つを残します。新しいstate保存が完了し、model・optimizer・scheduler・乱数・進捗ファイルが揃った後で古いstateを削除します。保存エラーや不完全な保存では前の正常なstateを残します。正常終了時のstate保存が成功した後は、それが最新の1つになります。別実行や再開元のstateを探して削除する処理はありません。既存の過去履歴の容量は自動では減りません。OFFにすれば保持制限を解除できます。

この方針はstateの保存回数ではなく保持数を減らします。保存時の書込み負荷と、一時的に新旧2つ分のディスク空き容量は必要です。古いstateを削除できなかった場合は警告をログに出し、学習を継続します。

### VRAM プリセット（sd-scripts）

| GPU VRAM | DiT / Qwen | CPU block swap |
|---|---|---|
| 12GB（初期値） | ConvRot INT8 / INT8 | 16 |
| 16GB | ConvRot INT8 / INT8 | 0 |
| 24GB 以上 | bf16 / bf16 | 0 |

いずれも batch 1、gradient checkpointing、latent / TE 出力のディスクキャッシュを ON にします。12GB でメモリ不足なら swap を 36 に上げたり、解像度を下げて調整できます。8GB 向けの動作は想定していません。必要 VRAM は画像・キャプション長などでも変わります。CPU スワップには十分な RAM とページファイルも必要です。

Attention は初期値の `torch` を推奨します。`xformers` を選ぶと必須の Split attention を自動で ON にし、変更不可にします。過去のプロジェクトで組み合わせが不整合でも読込時に補正します。`xformers` / `flash` 自体の追加パッケージは別途必要です。

### 生成用の初期値

「サンプル」タブに幅 1024 / 高さ 1024、50 ステップ、CFG 3.0、生成用 Flow shift 5.0、seed 42、negative 空欄、unconditional skip layer 22 を設定済みです。学習用 shift と生成用 shift は別項目です。

12GB では学習中の自動生成を初期 OFF にしています。ON にした場合は初期設定で 250 STEP ごとに生成します。プロンプトは学習対象に置き換えてください。1 行 1 件で、行内の `--w` / `--h` / `--s` / `--l` / `--fs` / `--d` / `--n` が共通生成値より優先されます。画像は学習バックエンドが実行フォルダの `models/sample` 配下に保存します。サンプルの TE 出力を学習開始時にキャッシュするため、実行中の変更は次回から反映されます。単独の推論アプリは含みません。

## プロジェクト

「＋ 追加 / コピー」で、推奨初期値からの新規作成と既存プロジェクトの設定コピーを選べます。デフォルト新規では現在のモデル設定（DiT・Qwen・VAE・Tokenizer cache・最大トークン数）を保持し、学習・生成設定は初期値、画像サブセットは空欄に戻します。コピー対象はモデルパス、画像パス、サブセット、学習・生成設定です。データ画像自体は複製しません。出力名、初期 LoRA 重み、再開 state 指定はリセットされ、実行履歴やモデルファイルはコピーされません。

設定は `data/projects/<ID>/project.json`、共通の Python / sd-scripts 設定は `data/environment.json` に保存します。`Ctrl+S` でも保存できます。保存した設定は GUI 再起動後も利用できます。`MOSS_TRAINER_DATA` 環境変数で保存先を変更できます。

### プロジェクトの削除

上部の「削除」で選択中のプロジェクトを一覧から外します。確認画面で承認すると、プロジェクトフォルダ全体を `data/projects/_trash/<ID>_<識別子>/` へ移動します。設定と実行履歴も移動し、未保存の編集は破棄します。プロジェクト外にあるモデル・画像の元ファイルはそのままです。モデルや画像をプロジェクトフォルダ内に置いていた場合、それらもフォルダと一緒に移動します。導入・学習・環境チェック中は追加・削除できません。

全プロジェクトを削除しても空の一覧を維持し、「追加 / コピー」から新しく作成できます。最後に削除したプロジェクトのモデル設定を新規作成時の引継ぎに使用します。削除を取り消す場合はアプリを終了し、退避したフォルダ内の `project.json` の `id` を確認し、そのフォルダを `data/projects/<ID>/` に戻してから起動してください。同じIDのフォルダが既にある場合は上書きしないでください。

### 設定の書き出し・読み込み

「設定を書き出す」で作られたフォルダには、再読み込み用の `project_snapshot.json` が入っています。「設定を読み込む」からこの JSON を選ぶと、設定・モデルパス・画像パスを確認したうえで、新しいプロジェクトとして取り込めます。保存済みプロジェクトの `project.json` や、実行履歴の `project_snapshot.json` も使用できます。

読み込み時は学習・生成設定、モデルと画像のパス、出力 LoRA 名を復元します。実行履歴は取り込まず、state 再開と初期 LoRA 重みの指定は解除します。このPCの Python / sd-scripts の共通設定を使用します。別PCのパスが存在しなくても取り込めるので、読み込み後に「参照…」で選び直してください。未対応の形式や不正な型は読み込み時にエラーを表示します。古い版で省略されている項目は初期値を補います。

対象は MOSS Studio が書き出す設定 JSON です。`train.toml` / `dataset.toml` を直接編集した内容を取り込む機能や、他GUIの設定形式の変換は含みません。

## 表示言語

ヘッダーの「日本語 / English」で切り替えます。選択した言語は `data/ui_preferences.json` に保存され、次回起動時にも引き継ぎます。未保存の設定、モデル・画像パス、プロンプト、既存プロジェクト名、ログの内容は保持されます。数値の入力途中など不正な値がある場合は、修正してから切り替えてください。導入・学習・環境チェック中は言語選択を無効にしています。

英語表示は項目名・説明・ダイアログ・アプリ側のエラー・進捗と終了予定表示が対象です。学習エンジンや導入子プロセスから届いたログは原文を表示します。Windows標準ダイアログのボタン等はOSの表示言語に従います。

## stateからの再開・追加学習

### 中断した学習を同じ計画で再開

1. 「学習」→「stateから再開・追加学習…」を開きます。
2. 保存済みstateを選び、「元のモデル・データ・学習設定を復元」をONにします。外部のstateはフォルダ選択できます。
3. 総STEP / Epoch上限を変えずに適用すると、保存したOptimizer・scheduler・累積STEPから元の計画を続けます。

### 最終stateから追加学習・総上限の変更

同じ画面で最終stateを選び、総上限を増やします。例: 保存STEPが1500で、総STEP上限を3000にすると、累積1500から再開して1500 STEP分を追加します。次の更新は1501、100 STEP保存なら次の保存は1600です。上限は追加分のSTEP数ではありません。Epoch指定のプロジェクトでは総Epoch上限を編集します。

上限変更時もOptimizerのモーメントや更新回数、乱数状態と累積STEPを復元します。ただし、LRスケジュールの計算には総STEP数が使われるため、**LRは新しい総上限の曲線上の保存STEP位置で再計算します**。constantは一定のまま、linear / cosineは再開時のLRが変わります。元の最終STEPでLRが0になっていても、上限を延ばすことで新しい曲線のLRで追加学習できます。Optimizerやschedulerを空回ししてSTEPを合わせる処理は行いません。保存schedulerの更新回数と `train_state.json` のSTEPが一致しない場合は、誤った補正を避けるため停止します。

元設定snapshotがある場合は、モデル・Rank・Alpha・Optimizer種類・scheduler種類・学習率・warmup・勾配累積・データ設定等を比較します。総上限、保存間隔、生成設定は変更できます。Optimizer種類や学習率等も変更する場合は、再開指定を解除し、LoRA重みから新規の追加学習を始めてください。元設定がない外部stateでは比較できないため、元の設定と環境を手動で揃えてください。

`skip_until_initial_step` は通常ONです。累積STEPとエポック内のバッチ位置を別々に扱い、勾配累積時も単位を揃えます。導入済みsd-scriptsのファイルは書き換えません。想定したソース構造と異なる版では、学習開始前にエラーで停止します。

再開は新しい実行フォルダを作り、`resume_source.json` に元stateのパスと進捗、`state_policy.json` にLR再計算とstate保持方針を記録します。以前の実行結果や再開元stateは上書きしません。「停止」は強制終了のため、未保存の進捗は失われます。直近の保存が完了したstateを選択してください。旧版で既にずれたSTEPのまま再保存したstateの履歴は、この補正だけでは復元できません。

### LoRA重みから新しいOptimizerで追加学習

「再開指定を解除」後に「初期 LoRA 重み」を指定します。この場合はOptimizerとschedulerを新規に作り、STEPを0から数えます。stateとの同時指定はできません。

## 実行結果・ログ

画面下部に学習STEPの進捗、残り時間、推定終了時刻を表示します。別の設定タブを開いていても確認できます。学習本体の `steps:` 形式のログから計算し、初期化・キャッシュ処理中は推定せず、速度が分かってから表示します。推定終了時刻はPCのローカル時刻です。

保存・サンプル生成・初回の最適化などで所要時間は変わるため、終了時刻は概算です。推定時刻を過ぎても新しい進捗がない場合は「計測・更新中」に戻ります。STEPが最後まで進んでも、モデルの保存や終了処理が終わるまでは「保存・終了処理中」と表示します。再開時のSTEP表示と推定は、学習バックエンドが報告する進捗を基準にします。

「実行・ログ」タブの「学習終了時に通知音」で通知音を切り替えられます。初期値はONで、学習の正常終了・失敗時に通知し、手動停止や環境導入時は鳴らしません。Windowsでは同梱の `assets/pokopin.wav`（約0.9秒の、柔らかい「ポ・コ・ピン♪」という3音）を使います。「音を試す」で確認できます。通知音のON/OFFはこのPC共通の設定として `data/ui_preferences.json` に保存します。再生音量はOS側の音量設定に従います。

実行ごとに `data/projects/<ID>/runs/<実行ID>/` を作成します。

```text
train.toml / dataset.toml      学習時に使った設定
sample_prompts.txt             自動生成 ON の場合
project_snapshot.json          開始時点の設定
environment_snapshot.json      Python / sd-scripts の指定
command.json                   実際に渡す引数配列
resume_source.json             state 再開の場合
training.log / status.json     全ログ・終了状態
models/                        LoRA・state・サンプル
tensorboard/                   TensorBoard ログ
```

「設定を書き出す」では実行せず TOML とスナップショットを別フォルダへ保存します。「現在の設定をプレビュー」で TOML とコマンドを確認できます。実行は shell 文字列ではなく引数配列を使います。Windows のプロセスは Job Object で管理し、停止時に Accelerate の子プロセスも終了します。画面のログは直近約 5000 行を保持し、ファイルには全ログを残します。

## 検証と範囲

```powershell
python -m unittest discover -s tests -v
```

RTX 4070（VRAM 12GB）で実機検証済みです（提供者による確認）。別途、GUI 起動、ファイル選択の呼び出し、プロジェクト作成・コピー、STEP/epoch 切替、プリセット、TOML、設定競合、state 候補・復元、ログ、テスト子プロセスの起動・停止を自動テストしています。導入処理は、新しい venv の実作成、模擬ダウンロードによる導入・修復経路、失敗時の参照パス維持、設定登録をテストしています。自動テストでは実モデルのGPU学習・生成・state復元を実行していません。詳細は [開発・検証](../CONTRIBUTING.md#検証状況) を参照してください。

## 参考資料

- [Anima Standalone Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer): プロジェクト管理とローカル trainer の参考。元コードを同梱・改変した fork ではありません。
- [MOSS-Image 学習ガイド](https://github.com/kohya-ss/sd-scripts/blob/main/docs/moss_image_train_network.md): MOSS の引数・推奨例・生成値・VRAM 設定の参考。
- [sd-scripts 学習引数](https://github.com/kohya-ss/sd-scripts/blob/main/library/args.py) / [train_network.py](https://github.com/kohya-ss/sd-scripts/blob/main/train_network.py): config と state 再開の仕様確認。

参照確認日: 2026-10-19。sd-scripts は利用者が指定する外部依存です。更新による変更は学習ログで確認してください。


## Kohakuバックエンド

詳細な設定値とモデル選びは [設定ガイド](STABLE_SETTINGS.md) を参照してください。以下はKohaku固有の手順と対応範囲です。

### 導入

1. `Start MOSS Studio.bat` を開き、右上の「かんたん導入」を押します。
2. 学習エンジンを `sd-scripts` または `kohaku` にします。
3. sd-scriptsはcu128（PyTorch 2.8）またはcu124（2.6）。Kohakuを選ぶとcu130（2.13）へ切り替わります。Kohakuには対応するNVIDIAドライバとVisual Studio Build ToolsのC++ワークロードが必要です。
4. 「導入・確認・修復を開始」を押します。専用環境だけを変更します。sd-scriptsは `backend/moss-env`、Kohakuは `backend/kohaku-env` です。
5. GUIの「04 VRAM・速度」で学習エンジンを選択します。環境を導入しただけでは既存プロジェクトのエンジンは変わりません。

Tritonは両環境へ自動導入します。Windowsでは `triton-windows` を使い、PyTorch 2.6→Triton 3.2、2.8→3.4、Kohakuの2.13→3.7に対応させています。`import triton` の確認に加え、GPU上で小さなカーネルをコンパイル・実行して正解と比較します。失敗した導入は成功として登録しません。

手動で登録したsd-scripts環境ではTritonがなくても従来の学習を続けられます。KohakuではTritonの実行テスト成功を必須にしています。Tritonの導入だけでINT8 backwardやtorch.compileを自動的にONにはしません。学習精度や速度は別途比較してください。

上流のAPI・依存バージョン変動を避けるため、Kohakuソースは `0c4737e3b8787a7df0f2d28a2972640a5f8522f6` に固定しています。


### 学習と保存

学習画像は指定フォルダ直下だけを列挙し、GUIと同じキャプション拡張子・複数行・class tokensの規則でJSONLを生成します。repeatも保持します。画像やキャプションの原本は変更しません。

実行順は設定チェック→テキストキャッシュ→Latentキャッシュ→学習です。いずれかが失敗すれば次の処理を実行しません。キャッシュはプロジェクト内の `kohaku-cache` に置き、異なるモデルのキャッシュを混ぜないようにします。

Kohakuは独自のMOSS objective、自然な縦横比に沿う解像度計画、LoRAターゲットのプリセットを使用します。sd-scriptsと同一の学習挙動・品質になるとは限りません。bucket刻み・最大最小解像度はKohakuの解像度計画へそのまま対応しません。

実行フォルダ:

- `kohaku_train.py` / `kohaku_data.py`: 自動生成した設定。
- `dataset.jsonl`: この実行で学習する画像とキャプション。
- `training.log`: GUIの実行ログ。
- `kohaku/metrics.jsonl`: 上流の計測ログ。
- `kohaku/checkpoints/step_XXXXXXXX`: Kohaku専用の再開checkpoint。
- `kohaku/export`: 正常終了時のComfyUI形式とLyCORIS形式のLoRA。
- `models`: GUIで指定した出力名を付けた最終LoRAと、保存時点のLoRAのコピー。古いstate整理の前にコピーするため、stateを最新1つにしても既存のLoRA出力は保持します。

再開ダイアログは両エンジンのcheckpointを検出します。sd-scriptsとKohakuのstateを相互に読み込むことはできません。


### 今回の接続で対応していない設定

Kohaku選択時は、Epoch指定、初期LoRA重みからの追加学習、Adafactor、モジュール別Rank/LR、Text embedder学習、Caption shuffle・色拡張・ランダムクロップ、Epoch単位の保存/生成、開始前サンプル生成、サンプル行内オプションを実行前に拒否します。周期LoRA保存はKohakuのcheckpoint exportを使うため、state保存も必要です。

constant以外のSchedulerでは、再開時に総STEPを変更する機能にも未対応です。元の総STEPを維持してください。

これらはKohaku本体の制限ではなく、このGUI接続の対応範囲です。


### 既存版からの更新

アプリを終了して新しいZIPを別フォルダへ展開し、必要に応じて旧版の `data` と `backend` を移してください。同じ場所へ上書きする場合、古い `start.bat` と `install.bat` はZIP展開だけでは削除されないため、手動で削除できます。新しい起動ファイルは `Start MOSS Studio.bat` の1つです。

ソース、LICENSE、THIRD_PARTY_NOTICES、`setup_backend.py` は必要です。`setup_backend.py` はGUIの導入処理に使います。


### ライセンス

KohakuMossTrainerは **KohakuMoss License 1.0** です。Apache-2.0に第10条の名称条件が追加された独自ライセンスであり、通常のApache-2.0として扱わないでください。

利用・改変・再配布が認められていますが、組み込んで配布する製品・パッケージ・主要ブランドのいずれかに **Kohaku** が必要です。この対応版の主要表示名を **MOSS Studio with Kohaku** にしています。非公開の内部ツールや、学習で生成したモデル・LoRA・画像などには第10条の適用除外があります。モデル本体の利用条件は別です。

配布時にはライセンス本文・著作権表示と、該当するNOTICEを保持します。Kohakuを改変して再配布する場合は変更の明記も必要です。このGUIは上流ソースを改変せず、専用venv内に取得したソースから実行します。取得したソースにはLICENSEとエンコーダーのNOTICEを残し、GUI側にも `licenses/KohakuMoss-LICENSE.txt` を収録しています。GUI独自部分のMITライセンスがKohakuへ適用されることはありません。

確認元: https://github.com/KohakuBlueleaf/KohakuMossTrainer/blob/0c4737e3b8787a7df0f2d28a2972640a5f8522f6/LICENSE


