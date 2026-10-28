"""Kohaku resume plans, usable by the GUI without importing torch.

The effective SCHEDULE is stored in each run's config.json. Reuse it on ordinary
resumes; extensions replace only the remaining cosine, starting at the saved LR.
"""
from contextlib import contextmanager
from copy import deepcopy
import math

from .i18n import tr


def factor_at(schedule, step, total):
    """Evaluate the supported constant/cosine AnySchedule recipes."""
    cfg = schedule.get('lr', schedule)
    mode = cfg['mode']
    start = cfg.get('start', 0)
    end = cfg.get('end', total)
    if end == -1:
        end = total
    warmup = cfg.get('warmup', 0)
    value, floor = cfg.get('value', 1.0), cfg.get('min_value', 0.0)
    if not all(math.isfinite(x) for x in (start, end, warmup, value, floor)) or end <= start + warmup:
        raise ValueError(tr('Kohaku: 保存された学習率スケジュールが不正です。'))
    if mode not in ('constant', 'cosine'):
        raise ValueError(tr('Kohaku: このスケジュールの延長には未対応です。元の終了STEPを使用してください。'))
    local = step - start
    if warmup and local < warmup:
        initial = cfg.get('init_value', 0.0) or 0.0
        return initial + (value - initial) * local / warmup
    if mode == 'constant':
        return value
    progress = min(1.0, max(0.0, (local - warmup) / (end - start - warmup)))
    return floor + (value - floor) * (1.0 + math.cos(math.pi * progress)) / 2.0


def resume_schedule(training, info, fallback):
    """Return the effective curve; reject ambiguous or unsafe plan changes."""
    source = info.get('snapshot')
    previous = source['settings']['training'] if source else None
    total = info.get('total_steps') or (previous['max_train_steps'] if previous else None)
    target, step = training['max_train_steps'], info['step']
    saved = info.get('schedule')
    mode = training['lr_scheduler']
    if total is None:
        if mode == 'cosine':
            raise ValueError(tr('Kohaku: Cosine再開には元のconfig.jsonが必要です。元の実行フォルダを保持してください。'))
        return deepcopy(fallback)
    if target < total:
        raise ValueError(tr('Kohaku: 再開時の終了STEPは元の終了STEP以上にしてください。'))
    if target == total:
        if mode == 'cosine' and not saved and previous and previous.get('resume'):
            raise ValueError(tr('Kohaku: Cosine再開には元のconfig.jsonが必要です。元の実行フォルダを保持してください。'))
        return deepcopy(saved or fallback)
    if mode == 'constant':
        return deepcopy(saved or fallback)
    if mode != 'cosine':
        raise ValueError(tr('Kohaku: 終了STEPの延長はconstantとcosineに対応しています。他の方式は元の終了STEPを使用してください。'))
    if not saved:
        raise ValueError(tr('Kohaku: Cosine再開には元のconfig.jsonが必要です。元の実行フォルダを保持してください。'))
    if step >= total:
        raise ValueError(tr('Kohaku: Cosineは完走済みです。途中のcheckpointを選択してください。'))
    cfg = saved.get('lr', saved)
    current = factor_at(saved, step, total)
    floor = cfg.get('min_value', 0.0)
    if current <= floor and step >= cfg.get('start', 0) + cfg.get('warmup', 0):
        raise ValueError(tr('Kohaku: 学習率が下限に達しています。下限に達する前のcheckpointを選択してください。'))
    if step < cfg.get('start', 0) + cfg.get('warmup', 0):
        # Finish the original warmup once, then decay over the longer duration.
        result = deepcopy(cfg)
        result['end'] = target
        return result
    # AnySchedule evaluates step 0 during construction. A positive start with
    # zero warmup divides by zero there. Encode the already-consumed prefix as
    # a flat ramp (identical endpoints); restored counters skip it entirely.
    return {'mode': 'cosine', 'end': target, 'warmup': step,
            'init_value': current, 'value': current, 'min_value': floor}


@contextmanager
def restore_schedule(schedule_class, *, saved_step, extending=False):
    """Load counters and optimizer state while retaining the effective curve.

    AnySchedule loads config/_last_lr but retains its constructed callables. Keep
    these consistent and check the extension against the actual checkpoint LR.
    Kohaku propagates restored LR to mirrored/fused optimizers in its usual path.
    """
    original = schedule_class.load_state_dict

    def load(schedule, state):
        if extending and state['last_epoch'] != saved_step:
            raise ValueError('Cannot extend: saved scheduler counter differs from checkpoint step')
        restored = {**state, 'config': deepcopy(schedule.config)}
        if extending:
            expected = [base * schedule.schedulers['lr'](state['last_epoch'])
                        for base in state['base_param_groups']['lr']]
            actual = state['_last_lr']
            if len(expected) != len(actual) or any(
                    not math.isfinite(a) or not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-12)
                    for a, b in zip(actual, expected)):
                raise ValueError('Cannot extend: saved learning rates do not match the resume plan')
        original(schedule, restored)
        schedule._last_lr = list(schedule.get_lr())
        for group, rate in zip(schedule.optimizer.param_groups, schedule._last_lr):
            group['lr'] = rate

    schedule_class.load_state_dict = load
    try:
        yield
    finally:
        schedule_class.load_state_dict = original
