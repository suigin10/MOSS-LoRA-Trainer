"""Estimate the end of training from its steps progress, excluding cache bars."""
from collections import deque
import math
import re
import time

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
STEPS = re.compile(r'\b(?:steps|train):\s*\d+%\|[^\r\n]*?\|\s*(\d+)/(\d+)\s*\[([^\]\r\n]*)\]')


def duration(seconds):
    seconds = max(0, math.ceil(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def clock_seconds(value):
    parts = value.strip().split(':')
    if len(parts) not in (2, 3) or any(not p.isdigit() for p in parts):
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


class TrainingProgress:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current = 0
        self.total = None
        self.deadline = None
        self.samples = deque(maxlen=20)
        self.first_measurement = None
        self.average_seconds = None

    def feed(self, text, now=None):
        now = time.monotonic() if now is None else now
        matches = list(STEPS.finditer(ANSI.sub('', text)))
        if not matches:
            return False
        match = matches[-1]
        current, total = int(match[1]), int(match[2])
        if total < 1 or current > total:
            return False
        if total != self.total or current < self.current:
            self.reset()
        self.current, self.total = current, total
        if not self.samples or current > self.samples[-1][1]:
            self.samples.append((now, current))
        details = match[3]
        # Use differences within this invocation, never cumulative resumed STEP.
        elapsed = clock_seconds(details.split('<', 1)[0])
        if elapsed is not None:
            if self.first_measurement is None:
                self.first_measurement = (current, elapsed)
            first_step, first_elapsed = self.first_measurement
            if current > first_step and elapsed > first_elapsed:
                self.average_seconds = (elapsed - first_elapsed) / (current - first_step)
        remaining = None
        if '<' in details:
            remaining = clock_seconds(details.split('<', 1)[1].split(',', 1)[0])
        if remaining is None:
            rate = re.search(r'(\d+(?:\.\d+)?)(s/it|it/s)', details)
            if rate and float(rate[1]) > 0:
                seconds_per_step = float(rate[1]) if rate[2] == 's/it' else 1 / float(rate[1])
                remaining = (total - current) * seconds_per_step
            elif len(self.samples) > 1:
                elapsed = now - self.samples[0][0]
                count = current - self.samples[0][1]
                if count > 0 and elapsed > 0:
                    remaining = (total - current) * elapsed / count
        if current >= total:
            remaining = 0
        self.deadline = None if remaining is None else now + remaining
        return True

    def remaining(self, now=None):
        if self.deadline is None:
            return None
        now = time.monotonic() if now is None else now
        value = self.deadline - now
        # A missed deadline means a sample/checkpoint/stall invalidated the last estimate.
        return value if value >= 0 else None
