"""UI translations. Templates are formatted before user data reaches a widget.

Context-local language keeps backend checks on worker threads independent.
Project values, paths, prompts, CLI choices and backend logs are never translated.
"""
from contextvars import ContextVar
import json
from pathlib import Path

_language = ContextVar('moss_ui_language', default='ja')
ENGLISH = json.loads((Path(__file__).with_name('locales') / 'en.json').read_text(encoding='utf-8'))


def set_language(language):
    _language.set(language if language in ('ja', 'en') else 'ja')


def tr(message, **values):
    template = ENGLISH.get(message, message) if _language.get() == 'en' else message
    return template.format(**values) if values else template
