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
