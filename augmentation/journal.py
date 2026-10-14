"""Durable private inference journal: exact cache identity and shared run budgets."""
import json
import os
from pathlib import Path
import time

from .review import digest
from model.runtime import write_json


class BudgetExhausted(RuntimeError): pass


class Journal:
    def __init__(self, output, recipe, identity, engine_factory, clock=time.monotonic, wall_clock=time.time):
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.recipe, self.identity, self.factory, self.clock = recipe, identity, engine_factory, clock
        self.wall_clock = wall_clock
        self.engine, self.role = None, None
        self.fingerprint = digest({'recipe': recipe, 'identity': identity})
        state_path = self.output / 'journal_state.json'
        state = json.loads(state_path.read_text()) if state_path.is_file() else {'fingerprint': self.fingerprint, 'elapsed_seconds': 0}
        if state['fingerprint'] != self.fingerprint: raise ValueError('Resume recipe/source/input identity differs')
        self.entries = {}
        self.path = self.output / 'calls.jsonl'
        for line in self.path.read_text().splitlines() if self.path.is_file() else []:
            row = json.loads(line)
            if row['output_sha256'] != digest(row['text']) or row['entry_sha256'] != digest({k: v for k, v in row.items() if k != 'entry_sha256'}):
                raise ValueError('Cached journal content/digest differs')
            key = (row['role'], row['key'])
            if key in self.entries: raise ValueError('Duplicate journal call key')
            self.entries[key] = row
        self.prior_elapsed = max([state['elapsed_seconds']] + [r['elapsed_seconds'] for r in self.entries.values()])
        if state.get('active'):
            # A terminated process cannot checkpoint its final monotonic time.
            # Charge the entire interval since its last durable checkpoint;
            # downtime after a hard interruption is conservatively included.
            self.prior_elapsed += max(0, wall_clock() - state['wall_checkpoint'])
        self.started = clock()
        self.attempts = max(state.get('attempts', len(self.entries)), len(self.entries))
        pending = self.output / 'inflight_call.json'
        if pending.is_file():
            request = json.loads(pending.read_text())
            if request['id'] not in {r['id'] for r in self.entries.values()}:
                with (self.output / 'interrupted_attempts.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(request, ensure_ascii=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
            pending.unlink()
        self._save_state()

    def elapsed(self): return self.prior_elapsed + self.clock() - self.started

    def check_budget(self):
        if self.attempts >= self.recipe['run']['max_calls'] or self.elapsed() >= self.recipe['run']['max_wall_minutes'] * 60:
            raise BudgetExhausted('Committed inference call/time budget exhausted')

    def call(self, role, key, prompt, sampling=False, *, response_schema=None):
        if role not in self.recipe['models'] or not isinstance(prompt, str): raise ValueError('Invalid inference role/prompt')
        options = dict(self.recipe['translation' if role == 'translator' else 'generation'])
        if response_schema is not None:
            options['response_schema'] = response_schema
        identity = digest({'role': role, 'key': key, 'prompt': prompt, 'sampling': sampling,
                           'options': options, 'model': self.recipe['models'][role], 'seed': self.recipe['seed']})
        previous = self.entries.get((role, key))
        if previous:
            if previous['id'] != identity: raise ValueError('Cached prompt/settings differ; do not reuse stale outputs')
            if previous['status'] == 'failed': raise RuntimeError('Prior generation failed; inspect the saved request/failure before resuming')
            return previous
        self.check_budget()
        if self.role != role:
            if self.engine: self.engine.close()
            self.engine, self.role = None, None
            self.engine = self.factory(self.recipe['models'][role], self.recipe['run']['device'])
            self.role = role
            if hasattr(self.engine, 'audit'): write_json(self.output / f'{role}_audit.json', self.engine.audit)
        self.check_budget()
        seed = (self.recipe['seed'] + int(identity[:8], 16)) % 2**31
        limited = {**options, 'max_time': min(180, max(1, self.recipe['run']['max_wall_minutes'] * 60 - self.elapsed()))}
        request = {'id': identity, 'role': role, 'key': key, 'prompt': prompt, 'sampling': sampling,
                   'model': self.recipe['models'][role], 'generation_seed': seed, 'effective_settings': limited,
                   'elapsed_seconds': self.elapsed()}
        self.attempts += 1
        self._save_state()
        write_json(self.output / 'inflight_call.json', request)
        status, error, failed = 'completed', None, None
        try: text = self.engine.generate(prompt, sampling, limited, seed)
        except ValueError as e: text, status, error = None, 'input_rejected', str(e)
        except Exception as e: text, status, error, failed = None, 'failed', str(e), e
        row = {'id': identity, 'role': role, 'key': key, 'prompt': prompt, 'prompt_sha256': digest(prompt),
               'sampling': sampling, 'settings': options, 'effective_settings': limited, 'generation_seed': seed, 'model': self.recipe['models'][role],
               'text': text, 'output_sha256': digest(text), 'status': status, 'error': error,
               'elapsed_seconds': self.elapsed(), 'index': len(self.entries)}
        row['entry_sha256'] = digest(row)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
        self.entries[role, key] = row
        self._save_state()
        (self.output / 'inflight_call.json').unlink()
        if failed: raise failed
        return row

    def _save_state(self, active=True):
        temporary = self.output / 'journal_state.tmp'
        write_json(temporary, {'fingerprint': self.fingerprint, 'elapsed_seconds': self.elapsed(), 'calls': len(self.entries), 'attempts': self.attempts,
                              'wall_checkpoint': self.wall_clock(), 'active': active})
        temporary.replace(self.output / 'journal_state.json')

    def close(self):
        if self.engine: self.engine.close(); self.engine = None
        self._save_state(active=False)

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
