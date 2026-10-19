"""Recalculate the LR at the saved step when the total training plan changes."""
import math


def realign_scheduler(wrapped, saved_step):
    scheduler = getattr(wrapped, 'scheduler', wrapped)
    if type(saved_step) is not int or saved_step < 0 or scheduler.last_epoch != saved_step:
        raise ValueError('Saved scheduler counter does not match train_state.json; use a consistent state checkpoint')
    functions = getattr(scheduler, 'lr_lambdas', None)
    if not functions or len(functions) != len(scheduler.base_lrs) or len(functions) != len(scheduler.optimizer.param_groups):
        raise ValueError('Changing the total training limit requires a supported LambdaLR scheduler')
    rates = [float(base * function(saved_step)) for base, function in zip(scheduler.base_lrs, functions)]
    if any(not math.isfinite(rate) or rate < 0 for rate in rates):
        raise ValueError('The extended LR schedule produced invalid learning rates')
    # Do not advance the optimizer or scheduler. The loaded optimizer moments and
    # scheduler counter are retained; only the LR at this point in the new curve changes.
    for group, rate in zip(scheduler.optimizer.param_groups, rates):
        group['lr'] = rate
    scheduler._last_lr = rates
    print(f'MOSS Studio: recalculated LR at saved step {saved_step} for the new training limit: {rates}', flush=True)
    return rates
