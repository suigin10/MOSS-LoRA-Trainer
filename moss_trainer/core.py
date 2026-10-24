from __future__ import annotations

from .i18n import tr
from .backends import backend, selected_environment, environment_valid, source_dir

import copy
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
from datetime import datetime
from uuid import uuid4

SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

# UI and serialization share this schema. CLI-only constants are in training_config.
# key: (Japanese label, default, optional choices, hint)
FIELDS = {
    'model': {
        'pretrained_model_name_or_path': ('MOSS-Image DiT', '', 'file', 'bf16 / ConvRot INT8。公式分割モデルは最初の shard を指定'),
        'text_encoder': ('Qwen3.5-9B', '', 'file', 'bf16 / ConvRot INT8。W4A8 GPTQ は非対応'),
        'vae': ('FLUX.2 VAE', '', 'file', '推奨: flux2-vae.safetensors'),
        'tokenizer_cache_dir': ('Tokenizer / HF cache', '', 'dir', '空欄: 初回に Hugging Face から自動取得'),
        'moss_max_token_length': ('キャプションの最大トークン数', 4096, None, 'テキストのみの上限。初期値4096。短くすると長い説明が切れます。画像との合計上限は04 VRAMのKohaku設定'),
    },
    'dataset': {
        'resolution': ('学習解像度', 1024, None, '正方形の基準解像度。16 の倍数'),
        'batch_size': ('バッチサイズ', 1, None, 'VRAM 12GB のビデオカード向け開始値は 1'),
        'enable_bucket': ('アスペクト比 bucket', True, None, '縦横比ごとに学習解像度を調整'),
        'bucket_reso_steps': ('Bucket 刻み', 64, None, '16 の倍数'),
        'min_bucket_reso': ('Bucket 最小解像度', 256, None, ''),
        'max_bucket_reso': ('Bucket 最大解像度', 2048, None, ''),
        'bucket_no_upscale': ('小さい画像を拡大しない', True, None, ''),
        'caption_extension': ('キャプション拡張子', '.txt', None, '画像と同じ名前の .txt を使用'),
        'enable_multiline_caption': ('複数行キャプション', True, None, '自然文キャプションの改行を保持'),
        'caption_dropout_rate': ('Caption dropout', 0.1, None, '自然文キャプションでは0推奨。Kohaku公式初期値0.1。TE キャッシュと併用可能'),
        'shuffle_caption': ('キャプションをシャッフル', False, None, 'TE キャッシュと併用不可。自然文では通常 OFF'),
        'color_aug': ('色の拡張', False, None, 'Latent キャッシュと併用不可'),
        'random_crop': ('ランダムクロップ', False, None, 'Latent キャッシュと併用不可'),
    },
    'training': {
        'output_name': ('出力 LoRA 名', 'moss_lora', None, 'safetensors のファイル名（拡張子不要）'),
        'network_algo': ('学習方式', 'LoRA', ['LoRA', 'LoKr', 'LoHa'], '両エンジンに対応。方式を変更するときはstateを外して新規学習してください'),
        'lokr_factor': ('LoKr factor', -1, None, 'LoKr専用。-1は自動分解、正の整数で分解係数を指定（例: 8）。Rankが同じでも方式によりパラメーター数が異なります'),
        'network_dim': ('Rank / dim', 16, None, 'このGUIの初期値16。方式・データセットに応じて調整'),
        'network_alpha': ('Alpha', 16.0, None, 'このGUIの初期値16。Rank・Alphaはstate再開時に変更できません'),
        'kohaku_train_target': ('Kohaku: 学習対象', 'ffn-down', ['ffn-down', 'attn-mlp', 'attn'], 'ffn-down: FFN_downのみ / attn-mlp: Attention＋FFN / attn: Attentionのみ。対象を変更するときは新規学習してください'),
        'learning_rate': ('学習率', 0.0001, None, '固定の最適値ではありません。データセットに応じて調整'),
        'optimizer_type': ('Optimizer', 'AdamW', ['AdamW', 'AdamW8bit', 'CAME', 'Adafactor'], '両エンジンの初期値はAdamW。CAMEは「かんたん導入」で必要パッケージを追加。Adafactorはsd-scripts専用'),
        'weight_decay': ('Weight decay', '', None, '空欄は各Optimizerの既定値。記事のキャラクターレシピは0.00005'),
        'kohaku_ema': ('Kohaku: EMA decay', 0.0, None, '0で無効。0より大きく1未満で有効（例: 0.99）。sd-scriptsでは適用されません'),
        'lr_scheduler': ('LR scheduler', 'cosine', ['constant', 'constant_with_warmup', 'cosine', 'cosine_with_restarts', 'linear', 'polynomial'], '初期値cosine。cosine_with_restartsは周期ごとに学習率を上げ直します。state再開では元の設定を維持'),
        'lr_scheduler_num_cycles': ('Restart周期数', 2, None, 'cosine_with_restarts専用。2は2周期（途中の再スタート1回）。warmupは学習開始時のみ'),
        'lr_scheduler_power': ('Polynomial power', 1.0, None, 'polynomial専用。1は直線的に減衰、1より大きいと早めに減衰。両エンジンで最終学習率は0'),
        'lr_warmup_steps': ('Warmup steps', 100, None, 'Kohaku公式初期値100。開始時に学習率を徐々に上げるstep数。constantでは使用しません'),
        'kohaku_lr_min_ratio': ('Kohaku: 最小学習率の比率', 0.1, None, 'cosine・cosine_with_restarts専用。0.1は初期学習率の10%。既存設定の未指定値は従来の0を維持'),
        'max_train_epochs': ('学習エポック数（steps=0 時）', 10, None, '最大 steps が 0 の場合だけ使用'),
        'max_train_steps': ('最大 steps（0: エポック指定）', 2500, None, 'このGUIの開始値2500。画像数・学習対象に応じて調整'),
        'gradient_accumulation_steps': ('勾配累積', 1, None, '実効バッチ数 = バッチサイズ × 勾配累積'),
        'max_grad_norm': ('勾配クリップ', 1.0, None, 'Adafactor 選択時は書き出し時に 0 を使用'),
        'save_every_n_epochs': ('何エポックごとに保存（0: 無効）', 0, None, ''),
        'save_every_n_steps': ('何 steps ごとに保存（0: 無効）', 100, None, 'このGUIの初期値100。LoRAと再開用stateを保存する間隔'),
        'save_state': ('再開用の optimizer state も保存', True, None, 'モデルの保存間隔に合わせて保存。ディスク使用量が増えます'),
        'keep_latest_state_only': ('state は最新の1つだけ保持', True, None, '新しいstateの保存成功後に、この実行で作成した古いstateだけを削除。LoRAは削除しません'),
        'save_state_on_train_end': ('正常終了時にも state を保存', True, None, '強制停止時には保存できません。直近の保存済み state から再開'),
        'skip_until_initial_step': ('再開時に処理済みバッチをスキップ', True, None, 'state の進捗とデータ位置を引き継ぐため通常 ON'),
        'seed': ('乱数 seed', 42, None, ''),
        'network_weights': ('初期 LoRA 重み（任意）', '', 'file', '重みからの追加学習。optimizer state の再開とは別'),
        'resume': ('再開 state フォルダ（任意）', '', 'dir', 'コピー作成時はリセット。通常の新規学習では空欄'),
    },
    'memory': {
        'backend': ('学習エンジン', 'kohaku', ['sd-scripts', 'kohaku'], 'Kohaku用環境は「かんたん導入」で別途作成。再開stateはエンジン間で共有できません'),
        'kohaku_memory_profile': ('Kohaku: メモリ構成', 'ckpt-resident', ['ckpt-ring4-actoff-optstream-gradoff', 'low', 'balanced', 'ckpt-resident', 'minimal', 'auto'], '12GBのINT8モデル向け初期値はckpt-resident。VRAM不足時はckpt-ring4-actoff-optstream-gradoffへ。下のswap・offload・Attentionはsd-scripts用'),
        'kohaku_base_precision': ('Kohaku: ベース計算', 'auto', ['auto', 'int8', 'fp8_compute'], 'autoはモデルの保存精度。FP8は対応GPUで実測比較してください'),
        'kohaku_max_tokens': ('Kohaku: 画像＋テキストのトークン上限', 6144, None, '12GB向け初期値6144。合計が超過するとテキストを切り詰めます。01モデルの上限はテキスト単体に適用'),
        'convrot_int8': ('DiT: ConvRot INT8', True, None, 'VRAM 12GB 向けの開始設定。量子化済みチェックポイントも自動検出されます'),
        'convrot_int8_text_encoder': ('Qwen: ConvRot INT8', True, None, 'VRAM 12GB 向けの開始設定。キャッシュ時の VRAM を削減'),
        'convrot_int8_bwd': ('INT8 backward', 'bf16', ['bf16', 'int8'], 'bf16 が高精度。int8 は Triton / CUDA が必要'),
        'blocks_to_swap': ('CPU にスワップするブロック数', 16, None, '0〜40。12GB向け推奨は16。余裕が足りない場合は20以上へ。画像・生成条件でも必要量は変わります'),
        'gradient_checkpointing': ('Gradient checkpointing', True, None, 'メモリ削減のため通常 ON'),
        'cpu_offload_checkpointing': ('CPU offload checkpointing', False, None, 'Block swap / Unsloth と併用不可'),
        'unsloth_offload_checkpointing': ('Unsloth offload checkpointing', False, None, 'Block swap / CPU offload と併用不可'),
        'cache_latents': ('Latent キャッシュ', True, None, ''),
        'cache_latents_to_disk': ('Latent をディスクに保存', True, None, ''),
        'cache_text_encoder_outputs': ('テキスト出力キャッシュ', True, None, 'Qwen は凍結し、キャッシュ後 GPU から解放'),
        'cache_text_encoder_outputs_to_disk': ('テキスト出力をディスクに保存', True, None, ''),
        'attn_mode': ('Attention', 'torch', ['torch', 'xformers', 'flash'], '通常は torch を推奨。xformers / flash は追加パッケージが必要'),
        'split_attn': ('Split attention', False, None, 'xformers 選択時は自動で ON になり、必須のため固定されます'),
        'max_data_loader_n_workers': ('DataLoader workers', 2, None, 'Windows 向けに少数の worker を使用'),
        'num_cpu_threads_per_process': ('CPU処理スレッド数', 2, None, 'DataLoader workersとは別設定で、増やしても速くなるとは限りません'),
        'persistent_data_loader_workers': ('Worker をエポック間で維持', True, None, 'workers=0 の場合は OFF'),
        'compile': ('torch.compile', False, None, '追加環境が必要。まず OFF で動作を確認'),
    },
    'flow': {
        'timestep_sampling': ('Timestep sampling', 'sigma', ['sigma', 'uniform', 'sigmoid', 'shift', 'flux_shift'], '初期値sigma。Kohakuはsigmaとshiftに対応。sigmaは従来のMOSS方式'),
        'weighting_scheme': ('Weighting scheme', 'logit_normal', ['logit_normal', 'uniform', 'sigma_sqrt', 'cosmap', 'none', 'mode'], ''),
        'logit_mean': ('Logit mean', -1.09, None, ''),
        'logit_std': ('Logit std', 0.95, None, ''),
        'discrete_flow_shift': ('学習用 flow shift', 1.0, None, 'sigmaの開始値1.0。shiftを試す場合は3.0が目安。サンプル生成用のshiftとは別'),
        'sigmoid_scale': ('Sigmoid scale', 1.0, None, ''),
        'mode_scale': ('Mode scale', 1.29, None, 'weighting_scheme=mode の場合に使用'),
        'train_text_embedder': ('DiT の Text embedder も学習', False, None, 'Qwen の学習とは別。TE キャッシュと併用可'),
        'embedder_lr': ('Text embedder の学習率', 0.00005, None, 'Text embedder ON かつモジュール別学習率が空欄のとき適用'),
        'exclude_patterns': ('除外パターン（1行1正規表現）', '', 'multiline', '例: .*ffn.*'),
        'include_patterns': ('包含パターン（1行1正規表現）', '', 'multiline', '除外指定より優先。re.fullmatch で一致'),
        'network_reg_dims': ('モジュール別 rank', '', None, r'例: .*proj\.[qkvo]=8,.*ffn.*=16'),
        'network_reg_lrs': ('モジュール別学習率', '', None, '例: .*text_embedder.*=5e-5（embedder 学習率より優先）'),
        'rank_dropout': ('Rank dropout', 0.0, None, ''),
        'module_dropout': ('Module dropout', 0.0, None, ''),
    },
    'samples': {
        'enabled': ('学習中にサンプル画像を生成', False, None, 'VRAM 12GB 向けの初期値は OFF。生成時にも VRAM を使用します'),
        'sample_every_n_epochs': ('何エポックごとに生成（0: 無効）', 0, None, ''),
        'sample_every_n_steps': ('何 steps ごとに生成（0: 無効）', 250, None, ''),
        'sample_at_first': ('学習前にも生成', False, None, ''),
        'width': ('生成画像の幅', 1024, None, '16 の倍数。VRAM が不足する場合は 768 / 512 へ'),
        'height': ('生成画像の高さ', 1024, None, '16 の倍数'),
        'infer_steps': ('生成ステップ数', 50, None, '公式パイプラインの初期値。速度優先なら 20〜30'),
        'cfg': ('CFG scale', 3.0, None, '公式パイプラインの初期値'),
        'flow_shift': ('生成用 flow shift', 5.0, None, '学習用 flow shift（初期値 1.0）とは別の設定'),
        'sample_seed': ('生成 seed', 42, None, '同じ seed と prompt で学習前後を比較'),
        'negative_prompt': ('Negative prompt', '', None, '公式の初期値は空欄'),
        'sample_uncond_skip_layers': ('Unconditional skip layers', '22', None, '空欄でスキップなし。0〜41 のカンマ区切り'),
        'prompts': ('プロンプト（1行1件）', 'A photo of a cat sitting by a window, soft natural light, detailed fur.', 'multiline', '学習対象に置き換えてください。改行は \\n と記述。行内の --w / --h / --s / --l / --fs / --d / --n は共通値より優先'),
    },
}
ENV_DEFAULTS = {'python': '', 'sd_scripts': '', 'kohaku_python': '', 'kohaku_source': ''}


def defaults(backend_name='kohaku'):
    settings = {group: {key: copy.deepcopy(spec[1]) for key, spec in fields.items()} for group, fields in FIELDS.items()}
    if backend_name == 'sd-scripts':
        # Historical fallback for saved projects/snapshots, not a backend-switch action.
        settings['memory']['backend'] = 'sd-scripts'
        settings['training'].update(network_dim=16, network_alpha=1.0, optimizer_type='AdamW8bit',
                                    lr_scheduler='constant', lr_warmup_steps=0, kohaku_lr_min_ratio=0.0,
                                    max_train_steps=1500, save_every_n_steps=100, kohaku_train_target='attn-mlp', learning_rate=0.0001)
        settings['dataset']['caption_dropout_rate'] = 0.05
        settings['flow'].update(timestep_sampling='sigma', discrete_flow_shift=1.0)
    return settings


def apply_preset(settings, preset):
    m = settings['memory']
    m.update(convrot_int8=preset != '24GB+', convrot_int8_text_encoder=preset != '24GB+',
             blocks_to_swap=16 if preset == '12GB' else 0,
             cpu_offload_checkpointing=False, unsloth_offload_checkpointing=False,
             gradient_checkpointing=True, cache_latents=True, cache_latents_to_disk=True,
             cache_text_encoder_outputs=True, cache_text_encoder_outputs_to_disk=True)
    settings['dataset']['batch_size'] = 1
    m['kohaku_memory_profile'] = 'ckpt-resident'


def apply_character_lokr_preset(settings):
    updates = {
        'training': {'network_algo': 'LoKr', 'lokr_factor': 8,
                     'optimizer_type': 'AdamW8bit', 'learning_rate': 0.0001,
                     'weight_decay': '0.00005', 'kohaku_ema': 0.99,
                     'gradient_accumulation_steps': 2, 'resume': '', 'network_weights': ''},
        'dataset': {'resolution': 512, 'batch_size': 2},
        'memory': {'convrot_int8': True, 'convrot_int8_text_encoder': True},
        'flow': {'timestep_sampling': 'shift', 'discrete_flow_shift': 3.0, 'sigmoid_scale': 1.0},
    }
    for group, values in updates.items():
        settings[group].update(values)
    return updates


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


class ProjectStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, project):
        pid = project['id']
        if not re.fullmatch(r'[a-f0-9]{32}', pid):
            raise ValueError(tr('プロジェクト ID が不正です'))
        return self.root / pid

    def create(self, name, source=None, *, preserve_output_name=False, model_settings=None):
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError(tr('名前は 1〜100 文字で入力してください'))
        if any(p['name'].casefold() == name.casefold() for p in self.list()):
            raise ValueError(tr('同じ名前のプロジェクトがあります'))
        settings = copy.deepcopy(source['settings']) if source else defaults()
        if source is None and model_settings is not None:
            # Keep the selected models while starting a fresh training plan.
            settings['model'] = {key: copy.deepcopy(model_settings.get(key, spec[1]))
                                 for key, spec in FIELDS['model'].items()}
        # Source paths are settings; run identity, state and results never carry over.
        settings['training'].update(resume='', network_weights='')
        if not preserve_output_name:
            settings['training']['output_name'] = 'moss_lora'
        project = {'schema_version': SCHEMA_VERSION, 'id': uuid4().hex, 'name': name,
                   'created_at': datetime.now().astimezone().isoformat(), 'settings': settings,
                   'subsets': copy.deepcopy(source['subsets']) if source else [{'image_dir': '', 'num_repeats': 1, 'class_tokens': ''}]}
        self.save(project)
        return project

    def save(self, project):
        atomic_json(self.directory(project) / 'project.json', project)

    def remove(self, project):
        source = self.directory(project)
        if source.resolve() != source or source.resolve().parent != self.root or not source.is_dir():
            raise ValueError(tr('プロジェクトの保存先が不正です'))
        trash = self.root / '_trash'
        if trash.resolve() != trash:
            raise ValueError(tr('削除済みプロジェクトの保存先が不正です'))
        trash.mkdir(exist_ok=True)
        destination = trash / (project['id'] + '_' + uuid4().hex[:12])
        source.rename(destination)
        return destination

    def load(self, path):
        p = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(p, dict) or p.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(tr('未対応のプロジェクト形式です'))
        if not isinstance(p.get('id'), str):
            raise ValueError(tr('プロジェクト ID が不正です'))
        self.directory(p)
        if not isinstance(p.get('name'), str) or not p['name'].strip() or len(p['name']) > 100 or not isinstance(p.get('subsets'), list) or not isinstance(p.get('settings'), dict):
            raise ValueError(tr('プロジェクト形式が不正です'))
        for sub in p['subsets']:
            if not isinstance(sub, dict) or set(sub) - {'image_dir', 'num_repeats', 'class_tokens'}:
                raise ValueError(tr('画像サブセットの形式が不正です'))
            if not isinstance(sub.get('image_dir'), str) or type(sub.get('num_repeats')) is not int or not isinstance(sub.get('class_tokens'), str):
                raise ValueError(tr('画像サブセットの型が不正です'))
        # Missing known fields get defaults; unknown fields are rejected, never silently launched.
        merged = defaults('sd-scripts')
        # A project saved before backend selection existed always used sd-scripts.
        merged['memory']['backend'] = p['settings'].get('memory', {}).get('backend', 'sd-scripts')
        for group, values in p['settings'].items():
            if group not in merged or not isinstance(values, dict):
                raise ValueError(tr('不明な設定: {p0}', p0=group))
            for key, value in values.items():
                if key not in merged[group]:
                    raise ValueError(tr('不明な設定: {p0}', p0=key))
                expected = type(merged[group][key])
                if type(value) is not expected and not (expected is float and type(value) is int):
                    raise ValueError(tr('設定の型が不正: {p0}', p0=key))
                if expected in (int, float) and not math.isfinite(value):
                    raise ValueError(tr('設定に非有限数は使えません: {p0}', p0=key))
                merged[group][key] = value
        p['settings'] = merged
        return p

    def list(self):
        result = []
        for path in self.root.glob('*/project.json'):
            # Corruption must be visible instead of silently hiding a project.
            result.append(self.load(path))
        return sorted(result, key=lambda p: p['name'].casefold())


def validate(project, env=None, files=False):
    s = project['settings']
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    for group, fields in FIELDS.items():
        for key, (_, default, choices, _) in fields.items():
            value = s[group][key]
            if type(default) in (int, float):
                require(type(value) in (int, float) and math.isfinite(value), tr('{p0}: 有限の数値が必要です', p0=key))
                if type(default) is int:
                    require(type(value) is int, tr('{p0}: 整数が必要です', p0=key))
            if isinstance(choices, list):
                require(value in choices, tr('{p0}: 選択肢が不正です', p0=key))
    if errors:
        return errors
    d, t, m, f = (s[k] for k in ('dataset', 'training', 'memory', 'flow'))
    is_sd_scripts = backend(s) == 'sd-scripts'
    for key in (('resolution', 'bucket_reso_steps', 'min_bucket_reso', 'max_bucket_reso') if is_sd_scripts else ('resolution',)):
        require(d[key] >= 16 and d[key] % 16 == 0, tr('{p0}: 16 以上の 16 の倍数にしてください', p0=key))
    if is_sd_scripts:
        require(d['min_bucket_reso'] <= d['max_bucket_reso'], tr('Bucket 最小値は最大値以下にしてください'))
        for key in ('min_bucket_reso', 'max_bucket_reso'):
            require(d[key] % d['bucket_reso_steps'] == 0, tr('{p0}: Bucket 刻みの倍数にしてください', p0=key))
    require(d['batch_size'] >= 1, tr('バッチサイズは 1 以上にしてください'))
    require(1 <= s['model']['moss_max_token_length'] <= 4096, tr('最大トークン数は 1〜4096 にしてください'))
    for key in ('network_dim', 'network_alpha', 'learning_rate', 'gradient_accumulation_steps'):
        require(t[key] > 0, tr('{p0}: 0 より大きい値が必要です', p0=key))
    if t['weight_decay'].strip():
        try:
            decay = float(t['weight_decay'])
            require(math.isfinite(decay) and decay >= 0, tr('Weight decayは0以上の有限の数値にしてください。'))
        except ValueError:
            require(False, tr('Weight decayは0以上の有限の数値にしてください。'))
    require(0 <= t['kohaku_ema'] < 1, tr('EMA decayは0以上1未満にしてください。'))
    if t['network_algo'] == 'LoKr':
        require(t['lokr_factor'] == -1 or t['lokr_factor'] > 0, tr('LoKr factorは-1または1以上の整数にしてください。'))
    if t['lr_scheduler'] == 'polynomial':
        require(t['lr_scheduler_power'] > 0, tr('Polynomial powerは0より大きい値にしてください。'))
    require(t['max_train_epochs'] >= 0, tr('{p0}: 0 以上にしてください', p0='max_train_epochs'))
    if t['max_train_steps'] == 0:
        require(t['max_train_epochs'] > 0, tr('{p0}: 0 より大きい値が必要です', p0='max_train_epochs'))
    if t['lr_scheduler'] == 'cosine_with_restarts':
        require(t['lr_scheduler_num_cycles'] > 0, tr('Restart周期数は1以上にしてください。'))
        if t['max_train_steps'] > 0:
            require(t['max_train_steps'] - t['lr_warmup_steps'] >= t['lr_scheduler_num_cycles'],
                    tr('再スタートにはwarmup後に周期数以上のSTEP数が必要です。'))
    for key in ('max_train_steps', 'lr_warmup_steps', 'save_every_n_steps', 'save_every_n_epochs', 'max_grad_norm', 'seed'):
        require(t[key] >= 0, tr('{p0}: 0 以上にしてください', p0=key))
    require(bool(re.fullmatch(r'[^<>:"/\\|?*\x00-\x1f]+', t['output_name'])) and t['output_name'] not in ('.', '..') and not t['output_name'].endswith((' ', '.')), tr('出力 LoRA 名に使用できない文字があります'))
    require(not re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?', t['output_name']), tr('Windows の予約名は出力名に使用できません'))
    require(d['caption_extension'].startswith('.') and not re.search(r'[/\\\s]', d['caption_extension']), tr('キャプション拡張子は .txt の形式で入力してください'))
    for group, key in (('training', 'kohaku_lr_min_ratio'), ('dataset', 'caption_dropout_rate'), ('flow', 'rank_dropout'), ('flow', 'module_dropout')):
        require(0 <= s[group][key] <= 1, tr('{p0}: 0〜1 にしてください', p0=key))
    if is_sd_scripts:
        require(0 <= m['blocks_to_swap'] <= 40, tr('Block swap は 0〜40 にしてください'))
        modes = sum([m['blocks_to_swap'] > 0, m['cpu_offload_checkpointing'], m['unsloth_offload_checkpointing']])
        require(modes <= 1, tr('Block swap / CPU offload / Unsloth はいずれか 1 つだけ選んでください'))
    require(not (m['cache_latents_to_disk'] and not m['cache_latents']), tr('ディスク Latent キャッシュには Latent キャッシュを ON にしてください'))
    require(not (m['cache_text_encoder_outputs_to_disk'] and not m['cache_text_encoder_outputs']), tr('ディスク TE キャッシュにはテキスト出力キャッシュを ON にしてください'))
    require(not (m['cache_latents'] and (d['color_aug'] or d['random_crop'])), tr('Latent キャッシュと色拡張 / ランダムクロップは併用できません'))
    require(not (m['cache_text_encoder_outputs'] and d['shuffle_caption']), tr('TE キャッシュとキャプションシャッフルは併用できません'))
    require(m['max_data_loader_n_workers'] >= 0, tr('DataLoader workers は 0 以上にしてください'))
    require(m['num_cpu_threads_per_process'] >= 1, tr('CPU処理スレッド数は1以上の整数にしてください'))
    if is_sd_scripts:
        require(not (m['max_data_loader_n_workers'] == 0 and m['persistent_data_loader_workers']), tr('workers=0 の場合は worker 維持を OFF にしてください'))
        require(m['attn_mode'] != 'xformers' or m['split_attn'], tr('xformers には Split attention が必要です'))
    for key in ('logit_std', 'discrete_flow_shift', 'sigmoid_scale', 'mode_scale', 'embedder_lr'):
        require(f[key] > 0, tr('{p0}: 0 より大きい値が必要です', p0=key))
    for key in ('exclude_patterns', 'include_patterns'):
        for pattern in f[key].splitlines():
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(tr('{p0}: 正規表現が不正: {p1}', p0=key, p1=exc))
    for key in ('network_reg_dims', 'network_reg_lrs'):
        for item in filter(None, f[key].split(',')):
            try:
                pattern, value = item.rsplit('=', 1)
                re.compile(pattern)
                number = int(value) if key.endswith('dims') else float(value)
                require(math.isfinite(number) and number > 0, tr('{p0}: 正数を指定してください', p0=key))
            except (ValueError, re.error):
                errors.append(tr('{p0}: pattern=値 の形式を確認してください', p0=key))
    if s['samples']['enabled']:
        require(any(line.strip() and not line.strip().startswith('#') for line in s['samples']['prompts'].splitlines()), tr('サンプルプロンプトを入力してください'))
        require(s['samples']['sample_every_n_epochs'] >= 0 and s['samples']['sample_every_n_steps'] >= 0 and (s['samples']['sample_every_n_epochs'] > 0 or s['samples']['sample_every_n_steps'] > 0), tr('サンプル生成間隔を steps または epoch で 1 以上にしてください'))
        layers = s['samples']['sample_uncond_skip_layers']
        require(not layers or all(x.strip().isdigit() and 0 <= int(x) <= 41 for x in layers.split(',')), tr('Skip layers は 0〜41 のカンマ区切りにしてください'))
    sample = s['samples']
    for key in ('width', 'height'):
        require(sample[key] >= 16 and sample[key] % 16 == 0, tr('生成 {p0}: 16 以上の 16 の倍数にしてください', p0=key))
    require(sample['infer_steps'] >= 1, tr('生成ステップ数は 1 以上にしてください'))
    require(sample['cfg'] >= 0 and sample['flow_shift'] > 0 and sample['sample_seed'] >= 0, tr('生成 CFG / seed は 0 以上、生成 shift は正数にしてください'))
    require(not re.search(r'[\r\n]|--', sample['negative_prompt']), tr('共通 Negative prompt に改行や -- は使用できません'))
    require(bool(project['subsets']), tr('画像サブセットを 1 件以上追加してください'))
    for i, sub in enumerate(project['subsets'], 1):
        require(type(sub.get('num_repeats')) is int and sub['num_repeats'] >= 1, tr('サブセット {p0}: repeat は 1 以上の整数です', p0=i))
        require(bool(sub.get('image_dir', '').strip()), tr('サブセット {p0}: 画像フォルダを指定してください', p0=i))
        if files and sub.get('image_dir'):
            path = Path(sub['image_dir'])
            require(path.is_absolute() and path.is_dir(), tr('画像フォルダが存在しません（絶対パス）: {p0}', p0=path))
            if path.is_dir():
                require(any(p.suffix.lower() in IMAGE_EXTENSIONS for p in path.iterdir() if p.is_file()), tr('画像がありません: {p0}（サブフォルダは自動再帰しません）', p0=path))
    if backend(s) == 'kohaku':
        from .kohaku_backend import problems
        errors.extend(problems(project))
    if files:
        env = selected_environment(env or ENV_DEFAULTS, s)
        require(Path(env['python']).is_file() and Path(env['python']).is_absolute(), tr('学習用 Python の実行ファイルを絶対パスで指定してください'))
        if is_sd_scripts and t['network_algo'] != 'LoRA':
            module = Path(source_dir(env)) / 'networks' / (t['network_algo'].lower() + '.py')
            base = module.with_name('network_base.py')
            require(module.is_file() and base.is_file() and 'MossImageBlock' in base.read_text(encoding='utf-8'),
                    tr('LoHa・LoKrのMOSS対応には新しいsd-scriptsが必要です。「かんたん導入」で別のフォルダに専用環境を作成してください。'))
        require(environment_valid(env), tr('選択した学習エンジンのPythonとソースを「かんたん導入」または「実行環境」で指定してください。'))
        for key in ('pretrained_model_name_or_path', 'text_encoder', 'vae'):
            path = Path(s['model'][key])
            require(bool(s['model'][key]) and path.is_absolute() and path.is_file(), tr('{p0}: モデルファイルを絶対パスで指定してください', p0=tr(FIELDS['model'][key][0])))
            match = re.search(r'-(\d{5})-of-(\d{5})\.safetensors$', path.name)
            if match:
                require(int(match[1]) == 1, tr('{p0}: 最初の shard を指定してください', p0=key))
                for n in range(1, int(match[2]) + 1):
                    shard = path.with_name(path.name[:match.start()] + f'-{n:05d}-of-{int(match[2]):05d}.safetensors')
                    require(shard.is_file(), tr('分割モデルが不足: {p0}', p0=shard.name))
        require('gptq' not in s['model']['text_encoder'].lower(), tr('GPTQ 形式の Qwen は非対応です。bf16 / ConvRot INT8 を指定してください'))
        for group, key in (('model', 'tokenizer_cache_dir'), ('training', 'resume'), ('training', 'network_weights')):
            value = s[group][key]
            require(not value or (Path(value).is_absolute() and (Path(value).is_file() if key == 'network_weights' else Path(value).is_dir())), tr('{p0}: パスが存在しません（絶対パス）', p0=key))
    if t['resume']:
        require(not t['network_weights'], tr('state からの再開と初期 LoRA 重みは同時に指定しないでください'))
        if files:
            info = state_info(t['resume'])
            errors.extend(info['errors'])
            if not info['errors']:
                if t['max_train_steps']:
                    require(t['max_train_steps'] > info['step'], tr('総 STEP 上限は保存済み {p0} steps より大きくしてください（追加 STEP 数ではありません）', p0=info['step']))
                else:
                    if info.get('final'):
                        require(t['max_train_epochs'] > info['epoch'], tr('最終stateの追加学習では総Epoch上限を保存済みEpochより大きくしてください'))
                    else:
                        require(t['max_train_epochs'] >= info['epoch'], tr('再開時の総エポック数は保存済み epoch {p0} 以上にしてください（同じエポックの途中から再開可能）', p0=info['epoch']))
                errors.extend(resume_differences(project, info))
    return errors


def state_info(folder):
    folder = Path(folder).resolve()
    if (folder / 'state.json').is_file():
        from .kohaku_backend import state_info as kohaku_state_info
        return kohaku_state_info(folder)
    info = {'backend': 'sd-scripts', 'path': str(folder), 'step': 0, 'epoch': 0, 'errors': [], 'snapshot': None, 'final': False}
    try:
        metadata = json.loads((folder / 'train_state.json').read_text(encoding='utf-8'))
        if type(metadata['current_step']) is not int or metadata['current_step'] < 0 or type(metadata['current_epoch']) is not int or metadata['current_epoch'] < 0:
            raise ValueError(tr('step / epoch が不正です'))
        info.update(step=metadata['current_step'], epoch=metadata['current_epoch'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        info['errors'].append(tr('state の進捗情報を読めません: {p0}', p0=exc))
    # Inspect structure only; do not deserialize pickle/optimizer files in the GUI.
    for label, patterns in [('LoRA model', ('model*.safetensors', 'pytorch_model*.bin')),
                            ('optimizer', ('optimizer*.bin',)), ('scheduler', ('scheduler*.bin',)),
                            (tr('乱数状態'), ('random_states_*.pkl',))]:
        try:
            found = any(p.is_file() and p.stat().st_size > 0 for pattern in patterns for p in folder.glob(pattern))
        except OSError:
            found = False
        if not found:
            info['errors'].append(tr('state の {p0} ファイルが不足 / 空です: {p1}', p0=label, p1=folder))
    snapshot = folder.parent.parent / 'project_snapshot.json'
    if snapshot.exists():
        try:
            saved = json.loads(snapshot.read_text(encoding='utf-8'))
            if saved.get('schema_version') != SCHEMA_VERSION or not isinstance(saved.get('settings'), dict) or not isinstance(saved.get('subsets'), list):
                raise ValueError(tr('未対応の snapshot 形式です'))
            for group, values in saved['settings'].items():
                if group not in FIELDS or not isinstance(values, dict) or any(key not in FIELDS[group] for key in values):
                    raise ValueError(tr('snapshot の設定形式が不正です'))
            info['snapshot'] = saved
            info['final'] = folder.name == saved['settings'].get('training', {}).get('output_name', 'moss_lora') + '-state'
        except (OSError, ValueError) as exc:
            info['errors'].append(tr('再開元の設定を読めません: {p0}', p0=exc))
    return info


def resume_differences(project, info):
    source = info.get('snapshot')
    if backend(project['settings']) != info.get('backend', 'sd-scripts'):
        return [tr('再開stateの学習エンジンが一致しません。元のエンジンを選択してください。')]
    if not source:
        return []
    differences = []
    for group, keys in {
        'model': tuple(FIELDS['model']), 'dataset': tuple(FIELDS['dataset']),
        'flow': tuple(FIELDS['flow']),
        'memory': ('backend', 'kohaku_memory_profile', 'kohaku_base_precision', 'kohaku_max_tokens'),
        'training': ('weight_decay', 'kohaku_ema', 'network_algo', 'lokr_factor', 'lr_scheduler_power', 'network_dim', 'network_alpha', 'kohaku_train_target', 'optimizer_type', 'lr_scheduler', 'learning_rate',
                     'lr_warmup_steps', 'lr_scheduler_num_cycles', 'kohaku_lr_min_ratio', 'gradient_accumulation_steps', 'max_grad_norm', 'seed'),
    }.items():
        for key in keys:
            if key == 'lokr_factor' and project['settings']['training']['network_algo'] != 'LoKr':
                continue
            if key == 'lr_scheduler_power' and project['settings']['training']['lr_scheduler'] != 'polynomial':
                continue
            if key in ('kohaku_train_target', 'kohaku_ema') and backend(project['settings']) != 'kohaku':
                continue
            if key == 'lr_scheduler_num_cycles' and project['settings']['training']['lr_scheduler'] != 'cosine_with_restarts':
                continue
            if key == 'kohaku_lr_min_ratio' and (backend(project['settings']) != 'kohaku' or project['settings']['training']['lr_scheduler'] not in ('cosine', 'cosine_with_restarts')):
                continue
            if (group == 'memory' and key == 'kohaku_memory_profile'
                    and backend(project['settings']) == 'kohaku'
                    and project['settings']['training']['optimizer_type'] == 'AdamW8bit'
                    and info.get('can_change_memory_profile')):
                # These profiles keep the same unfused bitsandbytes optimizer state.
                continue
            original = source['settings'].get(group, {}).get(key, 'sd-scripts' if key == 'backend' else 0.0 if key == 'kohaku_lr_min_ratio' else defaults('sd-scripts')[group][key])
            if project['settings'][group][key] != original:
                differences.append(tr('state 再開の設定差: {p0}。元設定を復元するか、LoRA 重みから追加学習してください', p0=tr(FIELDS[group][key][0])))
    if project['subsets'] != source['subsets']:
        differences.append(tr('state 再開の設定差: 画像サブセット。元の画像パス・repeat・class tokens に戻してください'))
    return differences


def resume_limit_changed(project, info):
    source = info.get('snapshot')
    if not source:
        return True
    before, after = source['settings']['training'], project['settings']['training']
    if before.get('max_train_steps', 1500) or after['max_train_steps']:
        return before.get('max_train_steps', 1500) != after['max_train_steps']
    return before.get('max_train_epochs', 10) != after['max_train_epochs']


def discover_states(project_dir):
    candidates = []
    for metadata in Path(project_dir).glob('runs/*/models/*/train_state.json'):
        info = state_info(metadata.parent)
        info['modified'] = metadata.stat().st_mtime
        candidates.append(info)
    for metadata in Path(project_dir).glob('runs/*/kohaku/checkpoints/step_*/state.json'):
        if metadata.parent.name.endswith('.partial'):
            continue
        info = state_info(metadata.parent)
        info['modified'] = metadata.stat().st_mtime
        candidates.append(info)
    return sorted(candidates, key=lambda x: (x['modified'], x['step']), reverse=True)


def toml_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, list)):
        return json.dumps(value, ensure_ascii=False)
    if not math.isfinite(value):
        raise ValueError(tr('TOML に非有限数は使用できません'))
    return str(value)


def dump_toml(values):
    return '\n'.join(f'{key} = {toml_value(value)}' for key, value in values.items()) + '\n'


def dataset_config(project):
    d = project['settings']['dataset']
    general_keys = ('caption_extension', 'enable_multiline_caption', 'shuffle_caption')
    dataset_keys = ('resolution', 'batch_size', 'enable_bucket', 'bucket_reso_steps', 'min_bucket_reso', 'max_bucket_reso', 'bucket_no_upscale')
    text = '[general]\n' + dump_toml({k: d[k] for k in general_keys})
    text += '\n[[datasets]]\n' + dump_toml({k: d[k] for k in dataset_keys})
    for sub in project['subsets']:
        text += '\n[[datasets.subsets]]\n' + dump_toml({**sub, 'caption_dropout_rate': d['caption_dropout_rate'], 'color_aug': d['color_aug'], 'random_crop': d['random_crop']})
    return text


def training_config(project, run_dir):
    s = project['settings']
    run_dir = Path(run_dir).resolve()
    config = {'network_module': 'networks.lora_generic', 'network_train_unet_only': True,
              'mixed_precision': 'bf16', 'save_precision': 'bf16', 'save_model_as': 'safetensors',
              'loss_type': 'l2', 'text_encoder_batch_size': 1,
              'output_dir': str(run_dir / 'models'), 'logging_dir': str(run_dir / 'tensorboard'),
              'log_with': 'tensorboard', 'dataset_config': str(run_dir / 'dataset.toml')}
    for group in ('model', 'training', 'memory'):
        config.update({k: v for k, v in s[group].items() if v != ''})
    for key in ('backend', 'kohaku_memory_profile', 'kohaku_base_precision', 'kohaku_max_tokens'):
        config.pop(key, None)
    config.pop('kohaku_lr_min_ratio', None)  # Kohaku-only scheduler parameter
    config.pop('kohaku_ema', None)
    weight_decay = config.pop('weight_decay', '')
    config.pop('kohaku_train_target', None)
    algo = config.pop('network_algo', 'LoRA')
    factor = config.pop('lokr_factor', -1)
    config['network_module'] = {'LoRA': 'networks.lora_generic', 'LoKr': 'networks.lokr', 'LoHa': 'networks.loha'}[algo]
    if config['lr_scheduler'] == 'polynomial':
        config['lr_scheduler_args'] = ['lr_end=0.0']
    else:
        config.pop('lr_scheduler_power', None)
    if config['lr_scheduler'] != 'cosine_with_restarts':
        config.pop('lr_scheduler_num_cycles', None)
    config.pop('keep_latest_state_only', None)
    config.pop('num_cpu_threads_per_process', None)  # accelerate launch option, not a trainer argument
    if not s['training']['resume']:
        config.pop('skip_until_initial_step', None)
    if config['max_train_steps'] > 0:
        config.pop('max_train_epochs')
    else:
        config.pop('max_train_steps')
    for key in ('save_every_n_steps', 'save_every_n_epochs'):
        if not config[key]:
            config.pop(key)
    if config['optimizer_type'] == 'Adafactor':
        config['optimizer_args'] = ['relative_step=False', 'scale_parameter=False', 'warmup_init=False']
        config['max_grad_norm'] = 0.0
    if weight_decay.strip():
        config.setdefault('optimizer_args', []).append(f'weight_decay={float(weight_decay)}')
    if config['optimizer_type'] == 'CAME':
        config['optimizer_type'] = 'pytorch_optimizer.CAME'
    if config['cpu_offload_checkpointing'] or config['unsloth_offload_checkpointing']:
        config['gradient_checkpointing'] = True
    f = s['flow']
    flow_keys = ('timestep_sampling', 'weighting_scheme', 'logit_mean', 'logit_std', 'discrete_flow_shift', 'sigmoid_scale', 'mode_scale')
    config.update({k: f[k] for k in flow_keys})
    args = [f'rank_dropout={f["rank_dropout"]}', f'module_dropout={f["module_dropout"]}']
    if algo == 'LoKr':
        args.append(f'factor={factor}')
    if f['train_text_embedder']:
        args.extend(['train_text_embedder=True'])
    for key in ('exclude_patterns', 'include_patterns'):
        patterns = [line.strip() for line in f[key].splitlines() if line.strip()]
        if patterns:
            args.append(f'{key}={patterns!r}')
    for key in ('network_reg_dims', 'network_reg_lrs'):
        value = f[key]
        if key == 'network_reg_lrs' and f['train_text_embedder'] and not value:
            value = f'.*text_embedder.*={f["embedder_lr"]}'
        if value:
            args.append(f'{key}={value}')
    config['network_args'] = args
    if s['samples']['enabled']:
        config.update({k: s['samples'][k] for k in ('sample_every_n_epochs', 'sample_every_n_steps', 'sample_at_first', 'sample_uncond_skip_layers')})
        for key in ('sample_every_n_epochs', 'sample_every_n_steps'):
            if not config[key]:
                config.pop(key)
        config['sample_prompts'] = str(run_dir / 'sample_prompts.txt')
    return config


def sample_prompts_text(settings):
    s = settings['samples']
    options = {'w': s['width'], 'h': s['height'], 's': s['infer_steps'], 'l': s['cfg'],
               'fs': s['flow_shift'], 'd': s['sample_seed']}
    if s['negative_prompt'].strip():
        options['n'] = s['negative_prompt'].strip()
    lines = []
    for line in s['prompts'].splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        suffix = ''.join(f' --{key} {value}' for key, value in options.items()
                         if not re.search(r'--' + key + r'(?:\s|$)', line))
        lines.append(line + suffix)
    return '\n'.join(lines) + '\n'


def command(env, run_dir, settings=None):
    env = selected_environment(env, settings or {})
    if env['backend'] == 'kohaku':
        from .kohaku_backend import command as kohaku_command
        return kohaku_command(env, run_dir, settings)
    threads = settings.get('memory', {}).get('num_cpu_threads_per_process', 2) if settings else 2
    if type(threads) is not int or threads < 1:
        raise ValueError(tr('CPU処理スレッド数は1以上の整数にしてください'))
    return [env['python'], '-u', '-m', 'accelerate.commands.launch', '--num_processes', '1',
            '--num_machines', '1', '--mixed_precision', 'bf16', '--dynamo_backend', 'no',
            '--num_cpu_threads_per_process', str(threads), str(Path(__file__).resolve().with_name('resume_bridge.py')),
            str(Path(env['sd_scripts']).resolve()),
            '--state-policy', str(Path(run_dir).resolve() / 'state_policy.json'),
            '--config_file', str(Path(run_dir).resolve() / 'train.toml')]


def write_run(project, env, run_dir):
    env = selected_environment(env, project['settings'])
    if env['backend'] == 'kohaku':
        from .kohaku_backend import write_configs
        errors = validate(project)
        if errors:
            raise ValueError('\n'.join(errors))
        Path(run_dir).mkdir(parents=True, exist_ok=False)
        return write_configs(project, env, run_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'dataset.toml').write_text(dataset_config(project), encoding='utf-8')
    (run_dir / 'train.toml').write_text(dump_toml(training_config(project, run_dir)), encoding='utf-8')
    if project['settings']['samples']['enabled']:
        (run_dir / 'sample_prompts.txt').write_text(sample_prompts_text(project['settings']), encoding='utf-8')
    atomic_json(run_dir / 'project_snapshot.json', project)
    atomic_json(run_dir / 'environment_snapshot.json', env)
    info = state_info(project['settings']['training']['resume']) if project['settings']['training']['resume'] else None
    atomic_json(run_dir / 'state_policy.json', {
        'keep_latest_state_only': project['settings']['training']['keep_latest_state_only'],
        'recalculate_scheduler': bool(info and resume_limit_changed(project, info))})
    argv = command(env, run_dir, project['settings'])
    atomic_json(run_dir / 'command.json', argv)
    if project['settings']['training']['resume']:
        atomic_json(run_dir / 'resume_source.json', {key: value for key, value in info.items() if key != 'snapshot'})
    return argv


def caption_audit(folder, extension='.txt', multiline=True):
    """Read captions using sd-scripts naming rules; never edit dataset files."""
    if not extension.startswith('.') or re.search(r'[/\\\s]', extension):
        raise ValueError(tr('キャプション拡張子は .txt の形式で入力してください'))
    result = {key: [] for key in ('images', 'matched', 'readable', 'missing', 'empty', 'encoding_errors', 'read_errors', 'name_hints')}
    folder = Path(folder)
    if not folder.is_dir():
        return result
    result['images'] = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name.casefold())
    for image in result['images']:
        base = str(image.with_suffix(''))
        candidates = [Path(base + extension)]
        tokens = base.split('_')
        if len(tokens) >= 5:
            candidates.append(Path('_'.join(tokens[:-4]) + extension))
        caption = next((p for p in candidates if p.is_file()), None)
        if caption is None:
            result['missing'].append((image, candidates[0]))
            # These names look plausible but are not read by the backend.
            for hint in (Path(str(image) + extension), Path(base + extension + extension)):
                if hint.is_file():
                    result['name_hints'].append((hint, candidates[0]))
            continue
        result['matched'].append((image, caption))
        try:
            text = caption.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            result['encoding_errors'].append(caption)
            continue
        except OSError as exc:
            result['read_errors'].append((caption, str(exc)))
            continue
        used_text = text if multiline else next(iter(text.splitlines()), '')
        if not used_text.strip().strip('\ufeff').strip():
            result['empty'].append(caption)
        else:
            result['readable'].append((image, caption))
    return result


def dataset_report(project):
    lines, total = [], 0
    ext = project['settings']['dataset']['caption_extension']
    for sub in project['subsets']:
        folder = Path(sub['image_dir'])
        if not sub['image_dir'].strip():
            lines.append(tr('画像フォルダが未指定です。画像そのものを含むフォルダを選択してください。'))
            continue
        if not folder.is_dir():
            lines.append(tr('画像フォルダが存在しません（絶対パス）: {p0}', p0=folder))
            continue
        audit = caption_audit(folder, ext, project['settings']['dataset']['enable_multiline_caption'])
        images, missing = audit['images'], len(audit['missing'])
        count = len(images) * sub['num_repeats']
        total += count
        lines.append(tr('{p0}: {p1} 枚 / caption なし {p2} 枚 / repeat 後 {p3} 枚', p0=folder, p1=len(images), p2=missing, p3=count))
        lines.append(tr('Caption 認識 {p0} / {p1} 件（同名ファイル {p2} 件 / 空の本文 {p3} 件 / UTF-8 エラー {p4} 件 / 読取エラー {p5} 件）',
                        p0=len(audit['readable']), p1=len(images), p2=len(audit['matched']), p3=len(audit['empty']), p4=len(audit['encoding_errors']), p5=len(audit['read_errors'])))
        lines.append(tr('検索設定: 拡張子 {p0} / 例: image.png → image{p0}（同じフォルダ）', p0=ext))
        for image, caption in audit['readable'][:3]:
            lines.append(tr('  認識: {p0} → {p1}', p0=image.name, p1=caption.name))
        for image, expected in audit['missing'][:10]:
            lines.append(tr('  Caption なし: {p0} / 探した名前: {p1}', p0=image.name, p1=expected.name))
        for actual, expected in audit['name_hints'][:10]:
            lines.append(tr('  名前を確認: {p0} は未対応です。期待する名前: {p1}', p0=actual.name, p1=expected.name))
        for caption in audit['empty'][:10]:
            lines.append(tr('  本文なし: {p0}（複数行OFFの場合は先頭行を使用）', p0=caption.name))
        for caption in audit['encoding_errors'][:10]:
            lines.append(tr('  UTF-8で保存し直してください: {p0}', p0=caption.name))
        for caption, error in audit['read_errors'][:10]:
            lines.append(tr('  読み取れません: {p0}: {p1}', p0=caption.name, p1=error))
        if not images:
            child_dirs = [p.name for p in folder.iterdir() if p.is_dir()][:10]
            if child_dirs:
                lines.append(tr('直下に画像がありません。子フォルダをサブセットとして追加してください: {p0}', p0=', '.join(child_dirs)))
    batch = project['settings']['dataset']['batch_size']
    accum = project['settings']['training']['gradient_accumulation_steps']
    if batch > 0 and accum > 0:
        lines.append(tr('合計 {p0} 枚相当 / 実効バッチ {p1} / 約 {p2} steps/epoch（単一 GPU）', p0=total, p1=batch * accum, p2=math.ceil(math.ceil(total / batch) / accum)))
    else:
        lines.append(tr('合計 {p0} 枚相当 / steps の概算には正のバッチサイズ・勾配累積が必要です', p0=total))
    lines.append(tr('caption がない画像には class tokens が使われます。画像フォルダは再帰走査しません。'))
    return '\n'.join(lines)


PROBE_CODE = '''import json, importlib.util
r={}
for m in ('torch','accelerate','safetensors','transformers','bitsandbytes','tensorboard','triton','xformers','flash_attn','kohakumoss','kohakuengine','pytorch_optimizer'):
 r[m]=importlib.util.find_spec(m) is not None
if r['torch']:
 import torch
 r['torch_version']=torch.__version__
 r['cuda']=torch.cuda.is_available()
 if r['cuda']:
  r['gpu']=torch.cuda.get_device_name(0)
  r['vram_gb']=round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)
  r['bf16']=torch.cuda.is_bf16_supported()
  r['compute_capability']=list(torch.cuda.get_device_capability(0))
print(json.dumps(r))
'''


def probe(env):
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    result = subprocess.run([env['python'], '-c', PROBE_CODE], cwd=source_dir(env), capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=90, creationflags=flags)
    if result.returncode:
        raise ValueError(result.stderr[-4000:] or tr('環境チェックに失敗しました'))
    info = json.loads(result.stdout.strip().splitlines()[-1])
    if info.get('triton') and info.get('cuda'):
        check = subprocess.run([env['python'], str(Path(__file__).with_name('triton_check.py'))],
                               cwd=source_dir(env), capture_output=True, text=True, encoding='utf-8',
                               errors='replace', timeout=180, creationflags=flags)
        info['triton_runtime'] = check.returncode == 0
        if check.returncode:
            info['triton_error'] = (check.stderr or check.stdout)[-2000:]
    return info


def probe_errors(info, settings):
    required = ['torch', 'accelerate', 'safetensors', 'transformers', 'tensorboard']
    kohaku = backend(settings) == 'kohaku'
    if kohaku:
        required = ['torch', 'safetensors', 'transformers', 'kohakumoss', 'kohakuengine', 'triton']
    if settings['training']['optimizer_type'] == 'AdamW8bit':
        required.append('bitsandbytes')
    if settings['training']['optimizer_type'] == 'CAME':
        required.append('pytorch_optimizer')
    if not kohaku and settings['memory']['attn_mode'] != 'torch':
        required.append({'xformers': 'xformers', 'flash': 'flash_attn'}[settings['memory']['attn_mode']])
    errors = [tr('{p0} が学習用 Python にインストールされていません', p0=m) for m in required if not info.get(m)]
    if not info.get('cuda'):
        errors.append(tr('CUDA GPU を使用できません'))
    elif not info.get('bf16'):
        errors.append(tr('選択した GPU / PyTorch は bf16 に対応していません'))
    if kohaku and not info.get('triton_runtime'):
        errors.append(tr('Kohaku: TritonのGPU実行テストに失敗しました。「かんたん導入」で修復してください。'))
    if kohaku and settings['memory']['kohaku_base_precision'] == 'fp8_compute' and tuple(info.get('compute_capability', (0, 0))) < (8, 9):
        errors.append(tr('Kohaku: FP8計算にはsm89以降のGPUが必要です。autoまたはint8を選択してください。'))
    if not kohaku and settings['memory']['convrot_int8_bwd'] == 'int8' and (not info.get('triton') or info.get('triton_runtime') is False):
        errors.append(tr('INT8 backward には Triton が必要です'))
    return errors


class WindowsJob:
    """Own the complete Accelerate process tree, including training workers."""
    def __init__(self, process):
        import ctypes
        from ctypes import wintypes as w
        class BasicLimits(ctypes.Structure):
            _fields_ = [('ProcessTime', ctypes.c_int64), ('JobTime', ctypes.c_int64),
                        ('Flags', w.DWORD), ('MinWorkingSet', ctypes.c_size_t),
                        ('MaxWorkingSet', ctypes.c_size_t), ('ActiveProcessLimit', w.DWORD),
                        ('Affinity', ctypes.c_size_t), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]
        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOps', 'WriteOps', 'OtherOps', 'ReadBytes', 'WriteBytes', 'OtherBytes')]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [('Basic', BasicLimits), ('Io', IoCounters),
                        ('ProcessMemory', ctypes.c_size_t), ('JobMemory', ctypes.c_size_t),
                        ('PeakProcessMemory', ctypes.c_size_t), ('PeakJobMemory', ctypes.c_size_t)]
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.kernel.SetInformationJobObject.restype = w.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = w.BOOL
        self.kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.kernel.TerminateJobObject.restype = w.BOOL
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.CloseHandle.restype = w.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = ExtendedLimits()
            limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            self.close()
            raise

    def terminate(self):
        if self.handle:
            self.kernel.TerminateJobObject(self.handle, 1)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class Runner:
    """One child process tree, durable logs, and UI-independent event delivery."""
    def __init__(self, emit):
        self.emit = emit
        self.process = None
        self.thread = None
        self.stopping = False
        self.job = None

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, argv, cwd, run_dir):
        if self.running:
            raise ValueError(tr('学習はすでに実行中です'))
        self.stopping = False
        self.thread = threading.Thread(target=self._run, args=(argv, cwd, Path(run_dir)), daemon=True)
        self.thread.start()

    def _run(self, argv, cwd, folder):
        started = datetime.now().astimezone().isoformat()
        status = {'status': 'running', 'started_at': started}
        try:
            atomic_json(folder / 'status.json', status)
            kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
            env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
            with (folder / 'training.log').open('w', encoding='utf-8', buffering=1) as log:
                child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding='utf-8', errors='replace', bufsize=1, **kwargs)
                try:
                    if os.name == 'nt':
                        self.job = WindowsJob(child)
                except Exception:
                    child.terminate()
                    child.wait()
                    child.stdout.close()
                    raise
                self.process = child
                if self.stopping:
                    self._terminate()
                for line in self.process.stdout:
                    log.write(line)
                    self.emit('log', line)
                code = self.process.wait()
                self.process.stdout.close()
            status.update(status='stopped' if self.stopping else ('completed' if code == 0 else 'failed'), returncode=code)
        except Exception as exc:
            status.update(status='failed', error=str(exc))
            self.emit('log', tr('\n起動エラー: {p0}\n', p0=exc))
        finally:
            if self.job:
                self.job.close()
                self.job = None
            status['finished_at'] = datetime.now().astimezone().isoformat()
            try:
                atomic_json(folder / 'status.json', status)
            finally:
                self.emit('done', status)

    def _terminate(self):
        p = self.process
        if p is None or p.poll() is not None:
            return
        if os.name == 'nt':
            if self.job:
                self.job.terminate()
        else:
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def stop(self):
        self.stopping = True
        threading.Thread(target=self._terminate, daemon=True).start()
