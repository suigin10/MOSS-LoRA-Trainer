# クレジットと外部コンポーネント

MOSS StudioのGUIと通知音はこのリポジトリの独自実装です。GUIの参考にしたプロジェクトのソースや、モデルの重みは同梱していません。

| プロジェクト | 利用方法 |
|---|---|
| [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | セットアップ時にソースを取得し、専用環境に導入する学習バックエンド |
| [Anima Standalone Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer) | GUI構成・プロジェクト管理の参考 |
| [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) | 学習対象のモデル・公式実装 |
| [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) | 推奨モデル案内からリンクする配布元 |
| [PyTorch](https://pytorch.org/) とsd-scriptsの依存ライブラリ | 専用venvへ別途インストール |

各コンポーネントの著作権・商標・ライセンスは配布元に帰属します。GUIのMITライセンスは、外部コンポーネントやモデルのライセンスを置き換えるものではありません。sd-scriptsのライセンス文書は、取得されたソース内で確認できます。

`assets/pokopin.wav` は本プロジェクト用に合成した通知音で、GUIと同じMITライセンスで配布します。外部の録音素材は使用していません。

## Kohaku対応版で追加した外部コンポーネント

- [KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer): 専用環境に取得する追加の学習バックエンド。KohakuMoss License 1.0（Apache-2.0ベース＋第10条の名称条件）。ライセンス本文は `licenses/KohakuMoss-LICENSE.txt`。主要表示名は **MOSS Studio with Kohaku**。上流ソースは改変せず、LICENSEおよび同梱NOTICEを保持します。
- [triton-windows](https://github.com/triton-lang/triton-windows): WindowsのGPUカーネル環境。MIT。専用venvへ別途インストールし、パッケージ付属のライセンスを保持します。
- Kohakuの依存ライブラリ（LyCORIS、KohakuEngine、AnySchedule、TileLangなど）は各パッケージのライセンスに従います。GUIのMITはこれらを置き換えません。

`assets/moss-studio*.png`、`assets/moss-studio.ico`、`assets/icons/*.png` はこのGUI向けの独自デザインで、GUIと同じMITライセンスです。既存ブランドのロゴ素材は使用していません。上流の公式アプリを示すものではありません。
