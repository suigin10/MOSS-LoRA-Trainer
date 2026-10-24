# 設定ガイド / Settings guide

## 学習の開始設定

| 項目 / Setting | Kohaku（新規初期値） | sd-scripts（手動で設定） |
| --- | --- | --- |
| 解像度 / Resolution | 1024 | 1024 |
| 学習方式 / Adapter type | LoRA | LoRA |
| Rank / Alpha | 16 / 16 | 16 / 16 |
| Optimizer | AdamW | AdamW |
| 学習率 / Learning rate | 0.0001 | 0.0001 |
| Scheduler | cosine | cosine |
| Warmup | 100 steps | 100 steps |
| 最小学習率の比率 / Minimum LR ratio | 0.1（初期学習率の10%） | 対象外 |
| 学習対象 / Training target | ffn-down（FFN_downのみ） | 標準（または下記のFFN_down指定） |
| 最大STEP / Total steps | 2500 | 2500 |
| 保存間隔 / Save interval | 100 steps | 100 steps |
| Caption dropout | 0.1（自然文は0推奨） | 0.05（自然文は0推奨） |
| Timestep sampling | sigma | sigma |
| 学習用 flow shift | 1.0 | 1.0 |
| Sigmoid scale | 1.0 | 1.0 |
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

## LoRA / LoKr / LoHa

「03 学習」の「学習方式」で選びます。Kohaku・sd-scriptsの両方に対応し、初期値はLoRAです。Rank・Alpha・学習率は方式を切り替えても保持します。同じRankでも学習パラメーター数や効き方は異なります。

| 方式 | 設定 |
| --- | --- |
| LoRA | 通常の低ランク方式。既存プロジェクトもLoRAを維持 |
| LoKr | Kronecker積を使う方式。factorの初期値-1は自動分解。参考ガイドのfactor 8も指定可能 |
| LoHa | Hadamard積を使う方式。LoKr factorは使用しません |

方式やLoKr factorを変える場合はstateを外して新規学習してください。sd-scriptsで初期重みを指定する場合も同じ方式の重みを使用します。古いsd-scriptsがLoHa・LoKrのMOSS対応を含まない場合は、「かんたん導入」で別フォルダに新しい専用環境を作成してください。

Kohakuの出力は従来どおりComfyUI形式・LyCORIS形式の2種類です。読み込み先にも選んだ方式の対応が必要です。LoHa・LoKrでAttentionを含めると、fused QKVのComfyUI出力に密行列の差分が含まれ、ファイルが大きくなる場合があります。

[参考ガイド](https://www.imagelora.dev/guides/how-to-train-moss-image-lora)のLoKr設定は別の学習ツールの例です。factor 8を全データセット共通の推奨値とはしていません。[sd-scripts公式](https://github.com/kohya-ss/sd-scripts/blob/main/docs/loha_lokr.md)

### キャラクターLoKr プリセット（実験的）

機能・設定値は実験的です。

「03 学習」の「キャラクターLoKr（実験的）」を押すと、[MOSS Image character recipe that works](https://www.imagelora.dev/guides/how-to-train-moss-image-lora)を参考に以下を設定します。「01 モデル」にはMOSS Image BaseのConvRot INT8モデルとQwenのConvRot INT8ファイルを指定してください。

| 項目 | 設定 |
| --- | --- |
| 学習方式 / LoKr factor | LoKr / 8 |
| Optimizer / 学習率 | AdamW8bit / 0.0001 |
| Weight decay | 0.00005 |
| Timestep sampling / 学習用flow shift / Sigmoid scale | shift / 3.0 / 1.0 |
| Kohaku: EMA decay | 0.99 |
| バッチ / 勾配累積 | 2 / 2（実効バッチ4） |
| 解像度 | 512 |
| DiT・Qwen ConvRot INT8 | ON |

適用すると再開state・初期重みの指定を解除します。設定を確認して保存してください。Kohakuでは通常版に加えて`-ema0.99-comfy.safetensors`・`-ema0.99-lycoris.safetensors`も出力し、再開stateにEMA状態を保存します。

## 学習対象

「03 学習」のKohaku学習対象は、ffn-down（FFN_downのみ、初期値）、attn-mlp（Attention＋FFN、従来設定）、attn（Attentionのみ）から選べます。ffn-downは全ブロックのFFN_downだけを学習します。画風LoRAの比較用に使えますが、効果はデータセットによって異なります。対象を変える場合は再開stateを外し、新規学習してください。対象欄のなかった既存プロジェクトは従来のattn-mlpを引き継ぎます。

sd-scriptsでもFFN_downのみを試す場合は「05 Flow・LoRA 詳細」で以下を入力します（Kohakuの対象欄は適用されません）。

- 除外パターン: `.*`
- 包含パターン: `layers[.][0-9]+[.]proj[.]ffn_down`

[参考: エマノン氏のMOSS-image LoRA作成まとめ](https://note.com/emanon_14/n/nc76e266c379f)。対象限定は学習条件の変更であり、画質が同一になる高速化ではありません。

## sd-scriptsの速度調整

- Latent・テキストのディスクキャッシュはONを維持します。
- VRAMに余裕があればblock swapを16から12、8と段階的に減らします。VRAM不足が出る場合は戻します。サンプル生成時の余裕も必要です。
- さらに余裕がある場合のみGradient checkpointingのOFFを比較します。VRAM使用量が大きく増えるため12GBではONから開始してください。
- 学習中サンプルをOFFにすると生成の待ち時間を減らせます。保存を減らす方法もありますが、再開ポイントが少なくなります。
- torch.compileは初回コンパイルの時間と追加メモリを含めて比較します。xformersへの変更やINT8 backwardへの変更で速くなるとは限りません。INT8 backwardは数値精度も変わります。

対応状況は[sd-scriptsのMOSS実装](https://github.com/kohya-ss/sd-scripts/blob/main/moss_image_train_network.py)と[包含・除外指定の実装](https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora_generic.py)を参照してください。

## Optimizer

両エンジンの新規初期値はAdamW、Schedulerはcosine・warmup 100です。AdamW8bitとCAMEも選べます。CAMEに切り替えても学習率・Schedulerは自動変更しません。CAMEはpytorch-optimizer 4.0.0を使い、専用環境へ「かんたん導入」で追加できます。KohakuではGPU上で通常stepを実行し、CAME標準のbetas=(0.9, 0.999, 0.9999)、eps1=1e-30、eps2=1e-16を使用します。Optimizerを変える場合はstateを外して新規学習してください。

## Schedulerの比較

既定はcosine（再スタートなし）です。比較するときは「03 学習」のLR schedulerでcosine_with_restartsを選びます。Kohaku・sd-scriptsの両方に対応します。

- Restart周期数の初期値は2です。2周期で、途中の再スタートは1回です。
- warmupは学習開始時だけに適用し、残りのSTEP数を周期に分けます。周期数はこのSchedulerを選んだときだけ編集できます。
- Kohakuは「最小学習率の比率」を各周期に適用します（初期値0.1）。sd-scriptsは各周期を0に向けて減衰します。割り切れない周期はKohakuでは整数STEPに分割します。
- Scheduler・周期数を変更するときはstateを外して新規学習してください。Kohakuのstate再開では総STEP数も維持します。

### Polynomial

両エンジンで`polynomial`も選択できます。Warmup後から0まで減衰します。専用の「Polynomial power」は1でlinear相当、1より大きいと早めに減衰し、0より大きく1未満では後半に急減します。初期値は1です。cosineやlinear等ではこの値を使用しません。

選択肢は`constant`・`constant_with_warmup`・`cosine`・`cosine_with_restarts`・`linear`・`polynomial`です。通常は既定のcosineから始めてください。`constant`ではwarmupを使用しません。

## Timestep sampling

新規Kohakuとsd-scriptsの開始設定はsigma・学習用flow shift 1.0です。sigmaではlogit mean -1.09・logit std 0.95を使用します。shiftを試す場合はTimestep samplingをshift、学習用flow shiftを3.0、sigmoid scaleを1.0に変更します。Kohakuではlogit_normal（mu=0、s=sigmoid scale）に静的shiftを適用し、sd-scriptsのshiftと同じ分布を作ります。KohakuのWeighting scheme欄はlogit_normalのまま使用します。shift時はlogit mean / stdを使用しません。

sigmaではWeighting schemeをlogit_normalにします。既存プロジェクトは保存済みのsigma・sigmoid等を維持します。方式を変更するときは新規学習を開始してください。サンプル生成用のflow shiftは別設定です。

## 再開・追加学習

- 既存プロジェクトは保存済み設定を保持します。新規初期値へ自動で置き換わりません。
- state再開では学習対象・Rank・Alpha・Optimizer・Scheduler・学習率などを元の設定に揃えてください。
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
