# 学習設定・モデルガイド / Training settings and models

## Kohakuの新規初期値

学習条件は [Kohaku公式のLoRA共通設定](https://github.com/KohakuBlueleaf/KohakuMossTrainer/blob/0c4737e3b8787a7df0f2d28a2972640a5f8522f6/configs/moss/_base/base_lora.py) に合わせています。
12GB向けのメモリ構成・batch・トークン量は、公式レシピとは分けてこのGUIの開始設定を使用します。
公式値はすべてのデータでの最適値を保証するものではありません。

Training defaults follow the upstream Kohaku LoRA recipe. Memory, batch size and token budget retain this GUI's 12GB starting configuration.

| 項目 / Setting | Kohaku新規初期値 | sd-scriptsの開始目安（手動で設定） |
| --- | --- | --- |
| 解像度 / Resolution | 1024 | 1024 |
| Rank / Alpha | 32 / 32 | 16 / 1 |
| Optimizer | AdamW | AdamW8bit |
| 学習率 / Learning rate | 1e-4 | 1e-4 |
| Scheduler / Warmup | cosine / 100 steps | constant / 0 |
| 最小学習率 / Minimum LR | 初期値の10%（1e-5） | constantのため固定1e-4 |
| 学習上限 / Total steps | 4000 | 1500（対象に応じて調整） |
| 保存間隔 / Save interval | 500 steps | 100 steps |
| Caption dropout | 0.1 | 0.05 |
| 勾配クリップ / Gradient clipping | 1 | 1 |
| Batch / 勾配累積 / Accumulation | 1 / 1 | 1 / 1 |
| DiT・Qwen | INT8モデルファイル | ConvRot INT8 ON |
| メモリ構成 / Memory | ckpt-resident | block swap 16 |
| ベース計算 / Base precision | auto | INT8 backward bf16 |
| キャプション上限（01モデル） | 4096 | 4096 |
| 画像＋テキスト上限（04 VRAM） | 6144 | 対象外 |
| Gradient checkpointing | ON（profile） | ON |
| CPU / Unsloth offload | profileに従う | OFF / OFF |
| Attention | 自動 / automatic | torch |
| Latent / text disk cache | ON / ON | ON / ON |
| DataLoader workers / CPU threads | 2 / 2 | 2 / 2 |
| torch.compile / 学習中サンプル | OFF / OFF | OFF / OFF |

Kohaku公式のメモリ構成balanced・token batching・MAX_TOKENS=12288はそのまま採用していません。
このGUIは12GB向けにckpt-resident・画像枚数でbatch 1・MAX_TOKENS=6144を使用し、seedは従来の42を維持しています。
新規初期値はKohaku用です。エンジンをsd-scriptsへ切り替えても学習条件を自動上書きしないため、右列の値を手動で設定してください。

Switching engines does not reset training settings. For sd-scripts, manually apply the right-hand column.

## 既存プロジェクトとstate再開

保存済みのRank・Alpha・Optimizer・Scheduler・保存間隔などは変更しません。
旧版のcosineは最小学習率0%だったため、新しい「Kohaku: 最小学習率の比率」が未指定の場合は0を補います。
新規プロジェクトは0.1です。state再開では最小学習率の比率も照合します。
既存の16/1のstateを32/32へ変更して再開することはできません。
現GUIのKohakuでは、constant以外のSchedulerで総STEP上限を変える追加学習は非対応です。cosineでは最初に総学習計画を決めてください。

Saved settings remain unchanged. Older cosine projects keep their 0% floor. New projects use 10%.
State resume checks the LR floor as well as rank, alpha, optimizer and scheduler.
Kohaku currently supports changing the total step limit on resume only with constant scheduling.

## 12GB向け設定と実測の範囲

INT8のDiTとQwenモデルを指定してください。autoは保存精度を維持し、bf16ファイルを自動でINT8に量子化しません。
VRAM不足時はckpt-ring4-actoff-optstream-gradoffへ変更できます。
AdamW8bitを選ぶ場合はGPU状態・fused_step=Falseを使用します。自動チューニングとtorch.compileはOFFです。

旧設定で1500STEPの学習完走を、利用者のRTX 4070 12GB実機で確認済みです（2026-10-21報告）。
従来のRank16/Alpha1・AdamW8bit・constantで、RTX 4070 12GBの短期診断は約4.10秒/step・約10.3GiBでした。
これは新しい32/32・AdamW・cosineの速度・VRAM・画質を測定した値ではありません。
torch.compileは従来構成で再計算エラーが発生したためOFFを維持します。

The user confirmed a completed 1500-step training run with the previous settings on their RTX 4070 12GB (reported 2026-10-21).
Previous short benchmark: about 4.10 s/step and 10.3 GiB on RTX 4070 12GB with rank16/alpha1, AdamW8bit and constant.
The new rank32/alpha32, AdamW, cosine recipe has not yet been benchmarked or quality-compared.

## VRAMプリセット

プリセットはメモリ関連項目とbatchのみ変更し、Rank・Optimizer・Schedulerは変更しません。
sd-scriptsの16GBはINT8 / swap 0、24GB以上はbf16 / swap 0が開始値です。
batch 1・checkpointing・disk cache ON。8GB向けプリセットはありません。

## 推奨モデルとダウンロード

INT8の12GB・16GB構成ではDiTにmoss_image_base_int8_convrot.safetensors、Qwenにqwen3.5_9b_int8_convrot.safetensors、VAEにflux2-vae.safetensorsを指定します。
24GB以上のbf16構成でQwenが分割されている場合は全4個を同じフォルダに置き、GUIにはmodel-00001-of-00004.safetensorsを指定します。
モデルファイルは任意のフォルダへ保存できます。既にComfyUIで使っているファイルも指定できます。

### ダウンロード先: 12GB / 16GB 共通

DiT（約5.41GB）:
https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_base_int8_convrot.safetensors

Qwen3.5-9B（約9.75GB）:
https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors

FLUX.2 VAE（全構成共通、約336MB）:
https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/vae/flux2-vae.safetensors

各ページの Download ボタンから取得できます。
記載容量はダウンロードファイルの容量で、学習時のVRAM使用量とは異なります。

### ダウンロード先: 24GB以上のbf16構成

DiT（約10.2GB）:
https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_base_bf16.safetensors

Qwen3.5-9B（4個合計で約16.8GB）:
https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/main/text_encoder

以下の4個をすべて取得してください:
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors

VAEは上記の flux2-vae.safetensors と共通です。

### SFT版を使う場合
学習・生成にSFT版を使いたい場合、DiTだけを次の対応ファイルに置き換えます。
QwenとVAEの指定方法は共通です。

12GB / 16GB用:
  moss_image_sft_int8_convrot.safetensors
  https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_sft_int8_convrot.safetensors

24GB以上のbf16構成用:
  moss_image_sft_bf16.safetensors
  https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_sft_bf16.safetensors

### 形式・設定について
・Qwenの w4a8_gptq 版は、このsd-scriptsの学習では非対応です。
  名前が似ていても qwen3.5_9b_int8_convrot.safetensors を選んでください。
・Tokenizer欄は通常空欄で構いません。初回は自動取得されます。
・キャッシュ、学習、サンプル生成で必要なメモリは異なります。
  実際の使用量は解像度・キャプション長などにも依存します。
・CPU block swapには十分なシステムRAMとページファイルが必要です。
・これは公式ガイドを参考にした開始用の推奨構成です。
  実機の確認環境と自動テストの範囲は、CONTRIBUTING.md の検証状況 に記載しています。

対応形式・VRAM設定の参考:
https://github.com/kohya-ss/sd-scripts/blob/main/docs/moss_image_train_network.md

Comfy-Orgの配布モデル一覧:
https://huggingface.co/Comfy-Org/MOSS-Image/tree/main
