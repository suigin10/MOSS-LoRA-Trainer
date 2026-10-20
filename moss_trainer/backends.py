"""Backend routing. Old environments and projects default to sd-scripts."""
from pathlib import Path


def backend(settings):
    return settings.get('memory', {}).get('backend', 'sd-scripts')


def selected_environment(env, settings):
    if backend(settings) == 'kohaku':
        return {**env, 'backend': 'kohaku',
                'python': env.get('kohaku_python', env.get('python', '') if env.get('backend') == 'kohaku' else ''),
                'source': env.get('kohaku_source', env.get('source', '') if env.get('backend') == 'kohaku' else '')}
    return {**env, 'backend': 'sd-scripts', 'python': env.get('python', ''), 'sd_scripts': env.get('sd_scripts', '')}


def source_dir(env):
    return env.get('source', env.get('kohaku_source', '')) if env.get('backend') == 'kohaku' else env.get('sd_scripts', '')


def environment_valid(env):
    source = Path(source_dir(env))
    script = 'scripts/train/moss.py' if env.get('backend') == 'kohaku' else 'moss_image_train_network.py'
    return (bool(env.get('python')) and Path(env['python']).is_absolute()
            and Path(env['python']).is_file() and source.is_absolute()
            and (source / script).is_file())
