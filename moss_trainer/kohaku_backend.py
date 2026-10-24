"""Generate Kohaku configs without importing GPU libraries into the GUI."""
import hashlib
import json
from pathlib import Path

from .i18n import tr


def training_preset(training):
    # Missing fields belong to projects created before target selection existed.
    target = training.get('kohaku_train_target', 'attn-mlp')
    if target == 'ffn-down':
        return {'target_name': [r'layers\.\d+\.proj\.ffn_down$']}
    if target in ('attn-mlp', 'attn'):
        return target
    raise ValueError(tr('Kohaku: 学習対象が不正です。'))


def timestep_knobs(flow):
    if flow['timestep_sampling'] == 'shift':
        # sd-scripts: u = sigmoid(sigmoid_scale * N(0,1)); sigma = shift(u).
        return 'logit_normal', {'mu': 0.0, 's': flow['sigmoid_scale'],
                                'shift': flow['discrete_flow_shift']}
    if flow['timestep_sampling'] == 'sigma':
        return 'moss', {'mu': flow['logit_mean'], 's': flow['logit_std'],
                        'shift': flow['discrete_flow_shift'], 'cutoff': 0.95, 'uniform_mix': 0.01}
    raise ValueError(tr('Kohaku: Timestepはsigmaまたはshift、Weighting schemeはlogit_normalを選択してください。'))

def problems(project):
    s = project['settings']
    t, m, d, f = (s[k] for k in ('training', 'memory', 'dataset', 'flow'))
    errors = []
    def need(ok, message):
        if not ok:
            errors.append(message)
    need(t['max_train_steps'] > 0, tr('Kohaku: STEP数を指定してください。Epoch指定には未対応です。'))
    need(not t['network_weights'], tr('Kohaku: 初期LoRA重みには未対応です。Kohakuのcheckpointから再開してください。'))
    need(t['optimizer_type'] in ('AdamW', 'AdamW8bit', 'CAME'), tr('Kohaku: OptimizerはAdamW・AdamW8bit・CAMEから選択してください。'))
    need(t['lr_scheduler'] in ('constant', 'constant_with_warmup', 'cosine', 'cosine_with_restarts', 'linear'), tr('Kohaku: Schedulerの設定を確認してください。'))
    need(t['save_every_n_epochs'] == 0, tr('Kohaku: 保存間隔はSTEP数で指定してください。'))
    need(t['save_state'] or t['save_every_n_steps'] == 0,
         tr('Kohaku: STEPごとのLoRA保存にはstate保存もONにしてください。'))
    need(m['cache_latents'] and m['cache_latents_to_disk'] and m['cache_text_encoder_outputs'] and m['cache_text_encoder_outputs_to_disk'],
         tr('Kohaku: LatentとテキストのディスクキャッシュをONにしてください。'))
    need(not d['shuffle_caption'] and not d['color_aug'] and not d['random_crop'],
         tr('Kohaku: Caption shuffle・色拡張・ランダムクロップをOFFにしてください。'))
    need(m['kohaku_max_tokens'] >= 1024, tr('Kohaku: トークン上限は1024以上にしてください。'))
    need(not f['train_text_embedder'] and not any(f[k] for k in ('exclude_patterns', 'include_patterns', 'network_reg_dims', 'network_reg_lrs')),
         tr('Kohaku: Text embedder学習とモジュール別の指定には未対応です。'))
    need(f['timestep_sampling'] in ('sigma', 'shift') and f['weighting_scheme'] == 'logit_normal',
         tr('Kohaku: Timestepはsigmaまたはshift、Weighting schemeはlogit_normalを選択してください。'))
    sample = s['samples']
    need(not sample['enabled'] or (sample['sample_every_n_steps'] > 0 and sample['sample_every_n_epochs'] == 0 and not sample['sample_at_first']),
         tr('Kohaku: サンプルはSTEP間隔で指定してください。開始前生成には未対応です。'))
    if sample['enabled']:
        need(not any(' --' in p for p in sample['prompts'].splitlines()),
             tr('Kohaku: サンプルの行内オプションには未対応です。共通パラメーターを使用してください。'))
    if t['resume']:
        info = state_info(Path(t['resume']))
        source = info.get('snapshot')
        if source and t['lr_scheduler'] != 'constant':
            need(t['max_train_steps'] == source['settings']['training']['max_train_steps'],
                 tr('Kohaku: constant以外のSchedulerでは再開時の総STEP変更に未対応です。元の総STEPを使用してください。'))
    return errors


def manifest(project):
    # Match the GUI's sd-scripts caption lookup, multiline setting and class fallback.
    from .core import caption_audit
    d = project['settings']['dataset']
    rows = []
    for sub in project['subsets']:
        audit = caption_audit(sub['image_dir'], d['caption_extension'], d['enable_multiline_caption'])
        if audit['encoding_errors'] or audit['read_errors']:
            raise ValueError(tr('Kohaku: CaptionのUTF-8エラーまたは読取エラーを修正してください。'))
        captions = dict(audit['matched'])
        for image in audit['images']:
            caption = sub['class_tokens']
            if image in captions:
                caption = captions[image].read_text(encoding='utf-8')
                if not d['enable_multiline_caption']:
                    caption = next(iter(caption.splitlines()), '')
            row = {'image': image.resolve().as_posix(), 'caption': caption}
            # Repetition is explicit so it does not become a loss multiplier.
            for _ in range(sub['num_repeats']):
                rows.append(row)
    if not rows:
        raise ValueError(tr('Kohaku: 学習画像がありません。'))
    return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)


def cache_directory(project, run_dir):
    identity = {}
    for key in ('pretrained_model_name_or_path', 'text_encoder', 'vae'):
        path = Path(project['settings']['model'][key])
        stat = path.stat() if path.is_file() else None
        identity[key] = (str(path.resolve()), stat.st_size if stat else 0, stat.st_mtime_ns if stat else 0)
    # Upstream caches include shape/caption keys. Separate model identities too.
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return Path(run_dir).parent.parent / 'kohaku-cache' / digest


def knobs(project, run_dir):
    s = project['settings']
    t, m, d, f, sample = (s[k] for k in ('training', 'memory', 'dataset', 'flow', 'samples'))
    run_dir = Path(run_dir).resolve()
    timestep, timestep_kwargs = timestep_knobs(f)
    schedule = {'mode': t['lr_scheduler'].replace('constant_with_warmup', 'constant'),
                'warmup': t['lr_warmup_steps'] if t['lr_scheduler'] != 'constant' else 0,
                'end': -1, 'min_value': t.get('kohaku_lr_min_ratio', 0.0) if t['lr_scheduler'] in ('cosine', 'cosine_with_restarts') else 0.0 if t['lr_scheduler'] == 'linear' else 1.0}
    if t['lr_scheduler'] == 'linear':
        schedule.update(mode='polynomial', power=1)
    if t['lr_scheduler'] == 'cosine_with_restarts':
        cycles = t.get('lr_scheduler_num_cycles', 2)
        warmup, total = t['lr_warmup_steps'], t['max_train_steps']
        if cycles < 1 or total - warmup < cycles:
            raise ValueError(tr('再スタートにはwarmup後に周期数以上のSTEP数が必要です。'))
        segments = []
        for i in range(cycles):
            segments.append({'mode': 'cosine', 'start': 0 if i == 0 else warmup + (total - warmup) * i // cycles,
                             'end': warmup + (total - warmup) * (i + 1) // cycles,
                             'warmup': warmup if i == 0 else 0, 'value': 1.0,
                             'min_value': schedule['min_value']})
        schedule = {'mode': 'composer', 'end': -1, 'schedules': segments}
    out = {
        'WEIGHTS': s['model']['pretrained_model_name_or_path'],
        'VARIANT': 'sft' if 'moss_image_sft' in Path(s['model']['pretrained_model_name_or_path']).name.lower() else 'base',
        'TEXT_ENCODER': s['model']['text_encoder'], 'VAE': s['model']['vae'],
        'DATASETS': [{'source': 'jsonl', 'path': (run_dir / 'dataset.jsonl').as_posix()}],
        'CAPTION': {'template': 'nl'}, 'CAPTION_VARIANTS': 1,
        'TARGET_AREA': d['resolution'] ** 2, 'RESOLUTION_MODE': 'resize' if d['enable_bucket'] else 'square_crop',
        'UPSCALE': 'never' if d['bucket_no_upscale'] else 'to_area',
        'BATCHING': 'count', 'BATCH_SIZE': d['batch_size'], 'MAX_TOKENS': m['kohaku_max_tokens'],
        'OVERLONG': 'truncate_text', 'MAX_TEXT_TOKENS': s['model']['moss_max_token_length'],
        'TEXT_DROP': d['caption_dropout_rate'], 'LATENT_CACHE': 'offline', 'TEXT_CACHE': 'offline',
        'CACHE_DIR': cache_directory(project, run_dir).as_posix(),
        'NUM_WORKERS': m['max_data_loader_n_workers'], 'DATA_SCAN_WORKERS': max(1, m['num_cpu_threads_per_process']),
        'SEED': t['seed'], 'ALGO': 'lora', 'PRESET': training_preset(t),
        'DIM': t['network_dim'], 'ALPHA': float(t['network_alpha']),
        'DROPOUT': {'input': 0.0, 'rank': f['rank_dropout'], 'module': f['module_dropout']},
        'TIMESTEP': timestep, 'TIMESTEP_KWARGS': timestep_kwargs,
        'OPTIMIZER': {'AdamW': 'adamw', 'AdamW8bit': 'adamw8bit', 'CAME': 'kohaku_train_entry.build_came'}[t['optimizer_type']],
        'LR': t['learning_rate'], 'SCHEDULE': schedule, 'GRAD_CLIP': t['max_grad_norm'],
        'PRECISION': 'bf16', 'MEMORY_PROFILE': m['kohaku_memory_profile'],
        # bitsandbytes owns its GPU states. Use its regular step after backward.
        'MEMORY_OVERRIDES': {'optim_state': 'gpu', 'fused_step': False} if t['optimizer_type'] in ('AdamW8bit', 'CAME') else {},
        'BASE_PRECISION': m['kohaku_base_precision'], 'FP8_DGRAD': '16bit',
        'TEXT_ENCODE': {'memory': 'ring', 'batch': 1, 'max_tokens': 8192, 'dtype': 'bf16'},
        'VAE_ENCODE': {'memory': 'offload', 'batch': 1, 'compile': 'off'}, 'VAE_DTYPE': 'bf16',
        # Stable Windows baseline: use kernels/tables without benchmark retuning.
        'TUNE': 'off', 'RETUNE_STEPS': 0,
        'COMPILE': 'block' if m['compile'] else 'off', 'STEPS': t['max_train_steps'],
        'GRAD_ACC': t['gradient_accumulation_steps'], 'OUTPUT_DIR': run_dir.as_posix(), 'RUN_NAME': 'kohaku',
        'LOG_EVERY': 1, 'METRICS_EVERY': 1, 'PROGRESS': True,
        'SAMPLE_EVERY': sample['sample_every_n_steps'] if sample['enabled'] else 0,
        'SAMPLE_PROMPTS': [p.strip() for p in sample['prompts'].splitlines() if p.strip() and not p.strip().startswith('#')] if sample['enabled'] else [],
        'SAMPLE_NEGATIVE': sample['negative_prompt'], 'SAMPLE_STEPS': sample['infer_steps'],
        'SAMPLE_CFG': sample['cfg'], 'SAMPLE_SHIFT': sample['flow_shift'],
        # Kohaku's sampling API takes (height, width), unlike the GUI fields.
        'SAMPLE_SIZE': (sample['height'], sample['width']), 'SAMPLE_SEEDS': (sample['sample_seed'],),
        'SAMPLE_SKIP_LAYERS': tuple(int(x) for x in sample['sample_uncond_skip_layers'].split(',') if x.strip()),
        'SAVE_EVERY': t['save_every_n_steps'] if t['save_state'] else 0,
        'KEEP_LAST': 1 if t['keep_latest_state_only'] else 0,
        'FINAL_CHECKPOINT': t['save_state_on_train_end'], 'RESUME': t['resume'],
        'SAVE_FORMAT': 'both', 'CHECKPOINT_EXPORT': True,
    }
    return out


def write_configs(project, env, run_dir):
    from .core import atomic_json
    errors = problems(project)
    if errors:
        raise ValueError('\n'.join(errors))
    run_dir = Path(run_dir).resolve()
    (run_dir / 'dataset.jsonl').write_text(manifest(project), encoding='utf-8')
    values = knobs(project, run_dir)
    config = '"""MOSS LoRA Trainer with Kohaku. Drives scripts/train/moss.py."""\n' + '\n'.join(f'{k} = {v!r}' for k, v in values.items()) + '\n'
    (run_dir / 'kohaku_train.py').write_text(config, encoding='utf-8')
    data = ('"""Drives scripts/data/cache_text.py and scripts/data/cache_latents.py."""\n'
            'from kohakuengine import use_config\n'
            f'use_config({str(run_dir / "kohaku_train.py")!r})\n'
            'TEXT_ENCODER = "kohakumoss.encoders.text_encoder"\n'
            f'TEXT_ENCODER_KWARGS = { {"path": values["TEXT_ENCODER"], "device": "cuda"}!r}\n'
            'IMAGE_ENCODER = "kohakumoss.encoders.vae_encoder"\n'
            f'IMAGE_ENCODER_KWARGS = { {"path": values["VAE"], "device": "cuda"}!r}\n'
            'ENCODE_BATCH = 1\n')
    (run_dir / 'kohaku_data.py').write_text(data, encoding='utf-8')
    atomic_json(run_dir / 'project_snapshot.json', project)
    atomic_json(run_dir / 'environment_snapshot.json', env)
    atomic_json(run_dir / 'kohaku_knobs.json', values)
    argv = command(env, run_dir, project['settings'])
    atomic_json(run_dir / 'command.json', argv)
    return argv


def command(env, run_dir, settings):
    bridge = Path(__file__).resolve().with_name('kohaku_bridge.py')
    return [env['python'], '-u', str(bridge), str(run_dir), str(env.get('source', env.get('kohaku_source', ''))),
            str(settings['memory']['num_cpu_threads_per_process'])]


def state_info(folder):
    info = {'path': str(folder), 'backend': 'kohaku', 'step': 0, 'epoch': 0,
            'errors': [], 'snapshot': None, 'final': False}
    try:
        saved = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
        if type(saved['step']) is not int or saved['step'] < 0:
            raise ValueError('Invalid STEP')
        if saved.get('world', 1) != 1:
            raise ValueError('This GUI supports single-GPU checkpoints only')
        info['step'] = saved['step']
        for name in ('trainable.safetensors', 'optim.pt', 'rng.0.pt'):
            if not (folder / name).is_file() or (folder / name).stat().st_size == 0:
                raise ValueError('Missing checkpoint file: ' + name)
        # Known GUI layout: <run>/kohaku/checkpoints/step_x; no recursive search.
        snapshot = folder.parent.parent.parent / 'project_snapshot.json'
        run_config = folder.parent.parent / 'config.json'
        if run_config.is_file():
            try:
                config = json.loads(run_config.read_text(encoding='utf-8'))
                memory = config.get('MEMORY_OVERRIDES', {})
                info['can_change_memory_profile'] = (config.get('OPTIMIZER') == 'adamw8bit'
                    and memory.get('optim_state') == 'gpu' and memory.get('fused_step') is False)
            except (OSError, ValueError, AttributeError):
                info['can_change_memory_profile'] = False
        if snapshot.is_file():
            from .core import FIELDS, SCHEMA_VERSION
            source = json.loads(snapshot.read_text(encoding='utf-8'))
            if source.get('schema_version') != SCHEMA_VERSION or not isinstance(source.get('settings'), dict) or not isinstance(source.get('subsets'), list):
                raise ValueError('Invalid project snapshot')
            for group, values in source['settings'].items():
                if group not in FIELDS or not isinstance(values, dict) or set(values) - FIELDS[group].keys():
                    raise ValueError('Invalid snapshot settings')
            info['snapshot'] = source
    except (OSError, ValueError, KeyError, TypeError) as exc:
        info['errors'].append(tr('Kohaku: checkpointを読めません: {p0}', p0=exc))
    return info
