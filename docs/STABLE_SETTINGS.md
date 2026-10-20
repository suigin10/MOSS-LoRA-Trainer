# 設定ガイド / Settings guide

## 学習の開始設定

| 項目 / Setting | Kohaku（新規初期値） | sd-scripts（手動で設定） |
| --- | --- | --- |
| 解像度 / Resolution | 1024 | 1024 |
| Rank / Alpha | 32 / 32 | 16 / 1 |
| Optimizer | AdamW8bit | AdamW8bit |
| 学習率 / Learning rate | 0.0001 | 0.0001 |
| Scheduler | cosine | constant |
| Warmup | 100 steps | 0 |
| 最小学習率の比率 / Minimum LR ratio | 0.1（初期学習率の10%） | 対象外 |
| 最大STEP / Total steps | 4000 | 1500 |
| 保存間隔 / Save interval | 200 steps | 100 steps |
| Caption dropout | 0.1 | 0.05 |
| 勾配クリップ / Gradient clipping | 1 | 1 |
| Batch / 勾配累積 / Accumulation | 1 / 1 | 1 / 1 |
| キャプション上限（01モデル） | 4096 | 4096 |
| 画像＋テキスト上限（04 VRAM） | 6144 | 対象外 |

STEP数・保存間隔は自由に変更できます。エンジンを切り替えても学習条件は自動上書きされないため、sd-scriptsでは右列を参考に設定してください。
Switching engines does not reset training settings. Apply the right-hand column manually for sd-scripts.

## 12GB向けのメモリ設定

| 項目 / Setting | Kohaku | sd-scripts |
| --- | --- | --- |
| DiT・Qwen | INT8モデルファイル | ConvRot INT8 ON |
| メモリ構成 / Memory profile | ckpt-resident | block swap 16 |
| ベース計算 / Base precision | auto | INT8 backward bf16 |
| Gradient checkpointing | profileに従う | ON |
| CPU / Unsloth offload | profileに従う | OFF / OFF |
| Attention | 自動 | torch |
| Latent / text disk cache | ON / ON | ON / ON |
| DataLoader workers / CPU threads | 2 / 2 | 2 / 2 |
| torch.compile / 学習中サンプル | OFF / OFF | OFF / OFF |

- Kohakuのautoはモデルの保存精度を使います。12GBではINT8のDiT・Qwenファイルを指定してください。
- KohakuでVRAMが不足する場合は、メモリ構成をckpt-ring4-actoff-optstream-gradoffへ変更します。
- VRAMプリセットはメモリ関連項目とbatchのみ変更します。sd-scriptsの16GBはINT8 / swap 0、24GB以上はbf16 / swap 0が開始値です。

## 再開・追加学習

- 既存プロジェクトは保存済み設定を保持します。新規初期値へ自動で置き換わりません。
- state再開ではRank・Alpha・Optimizer・Scheduler・学習率などを元の設定に揃えてください。
- Kohakuのcosine等では再開時に総STEP上限を変更できません。総上限を増やす追加学習はconstantで対応します。
- 旧版のcosine設定で最小学習率の比率が未指定の場合は、従来の0を引き継ぎます。

## モデルの指定

| 用途 | ファイル・ダウンロード先 |
| --- | --- |
| INT8 DiT（12GB / 16GB） | [moss_image_base_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_base_int8_convrot.safetensors) |
| INT8 Qwen（12GB / 16GB） | [qwen3.5_9b_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors) |
| VAE（共通） | [flux2-vae.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/vae/flux2-vae.safetensors) |
| bf16 DiT（24GB以上） | [moss_image_base_bf16.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_base_bf16.safetensors) |
| bf16 Qwen（24GB以上） | [MOSS-Image-Base / text_encoder](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/main/text_encoder) |

bf16 Qwenの分割ファイル4個は同じフォルダに置き、GUIにはmodel-00001-of-00004.safetensorsを指定します。
Tokenizer欄は通常空欄で構いません。QwenのW4A8 GPTQ形式は非対応です。
モデルは任意のフォルダに保存でき、ComfyUIで使っているファイルも指定できます。

SFT版を使う場合はDiTを [INT8 SFT](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_sft_int8_convrot.safetensors) または [bf16 SFT](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_sft_bf16.safetensors) に変更します。
