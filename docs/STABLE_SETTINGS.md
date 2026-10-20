# 12GB向け設定の目安 / Starting settings for 12GB

新規学習の開始設定です。データに応じて調整してください。
Starting values for new training; adjust to your dataset.

| 項目 / Setting | Kohaku | sd-scripts |
| --- | --- | --- |
| 解像度 / Resolution | 1024 | 1024 |
| Batch / 勾配累積 / Accumulation | 1 / 1 | 1 / 1 |
| Rank / Alpha | 16 / 1 | 16 / 1 |
| Optimizer / LR | AdamW8bit / 1e-4 | AdamW8bit / 1e-4 |
| Scheduler / Warmup | constant / 0 | constant / 0 |
| 勾配クリップ / Grad clip | 1 | 1 |
| DiT・Qwen / DiT and Qwen | INT8モデルファイル / INT8 model files | ConvRot INT8 ON |
| メモリ構成 / Memory profile | ckpt-resident | block swap 16 |
| ベース計算 / Base precision | auto | INT8 backward bf16 |
| Token limit | 6144 (画像＋テキスト / image + text) | text 4096 |
| Gradient checkpointing | ON (profile) | ON |
| CPU / Unsloth offload | profile | OFF / OFF |
| Attention | 自動 / automatic | torch |
| Latent・text disk cache | ON / ON | ON / ON |
| DataLoader workers / CPU threads | 2 / 2 | 2 / 2 |
| torch.compile | OFF | OFF |
| 学習中サンプル / Training samples | OFF | OFF |

## Kohaku

torch.compileは現環境のINT8・ckpt-residentで再計算エラーを確認したためOFFを維持します。詳細は [開発・検証](../CONTRIBUTING.md#検証状況) を参照。


RTX 4070 12GBの各12STEP短期比較で、従来の省メモリ構成は4.518秒/step、
ckpt-residentは4.102秒/step、デバイスVRAMは約10.3GiBでした。
画像・キャプションによって必要VRAMと速度は変わります。
VRAM不足時は ckpt-ring4-actoff-optstream-gradoff を選択してください。
autoは保存精度を維持します。bf16モデルファイルを自動でINT8に量子化しません。
AdamW8bitの状態はGPU、fused_step=False、自動チューニングOFFです。

Short RTX 4070 12GB benchmark: 4.518 s/step with the previous offload profile,
4.102 s/step and about 10.3 GiB device VRAM with ckpt-resident.
Memory and speed depend on images and captions. If VRAM runs out, use
ckpt-ring4-actoff-optstream-gradoff. auto preserves stored precision; it does not
quantize bf16 model files to INT8. AdamW8bit states stay on GPU, fused_step=False,
and automatic tuning is disabled.

## プリセット / Presets

既存プロジェクトの保存値は自動変更しません。プリセットボタンはVRAM関連項目とbatchのみ変更します。
sd-scriptsの16GBはINT8 / swap 0、24GB以上はbf16 / swap 0が開始値です。
どちらもbatch 1・checkpointing・disk cache ON。8GB向けプリセットはありません。

Existing saved projects are preserved. Preset buttons change VRAM-related options
and batch size only. sd-scripts starts with INT8 / swap 0 on 16GB and bf16 / swap 0
on 24GB+, both with batch 1, checkpointing and disk caches ON. No 8GB preset.


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
