"""One spawned process per native model; exiting releases its CUDA context."""
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile

from .inference import NativeEngine


def gpu_memory(device):
    if device != 'cuda': return {}
    import torch
    result = {}
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        result[f'cuda:{index}'] = {'free_bytes': free, 'total_bytes': total}
    return result


def _serve(connection, settings, device, factory):
    engine = None
    try:
        initial_memory = gpu_memory(device)
        engine = factory(settings, device)
        connection.send({'kind': 'ready', 'audit': {**getattr(engine, 'audit', {}),
            'worker_pid': os.getpid(), 'execution': 'isolated_spawn_process',
            'memory_at_worker_start': initial_memory}, 'cache': getattr(getattr(engine, 'cache', None), 'name', None)})
        while True:
            request = connection.recv()
            if request['operation'] == 'close': break
            try:
                text = engine.generate(request['prompt'], request['sampling'], request['options'], request['seed'])
                connection.send({'kind': 'output', 'text': text})
            except Exception as error:
                connection.send({'kind': 'error', 'type': type(error).__name__, 'error': str(error)})
    except EOFError:
        pass
    except BaseException as error:
        try: connection.send({'kind': 'error', 'type': type(error).__name__, 'error': str(error)})
        except (BrokenPipeError, EOFError, OSError): pass
    finally:
        if engine: engine.close()
        connection.close()


class ProcessEngine:
    """The parent owns prompts/journals; the worker owns all native weights."""
    def __init__(self, settings, device, engine_factory=NativeEngine):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.device, self.cache = device, None
        self.closed = False
        self.process = context.Process(target=_serve, args=(child, settings, device, engine_factory), daemon=True)
        try:
            self.process.start(); child.close()
            response = self._receive()
            if response['kind'] != 'ready': raise RuntimeError('Inference worker did not finish initialization')
            self.audit, self.cache = response['audit'], response['cache']
            print(json.dumps({'augmentation_worker_started': self.audit['worker_pid'], 'model': settings.get('name'),
                              'memory_at_worker_start': self.audit['memory_at_worker_start']}), flush=True)
        except BaseException:
            child.close(); self.close(); raise

    def _receive(self):
        try: response = self.connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError('Inference worker exited unexpectedly; inspect its run log') from error
        if response['kind'] == 'error':
            message = response['type'] + ' in inference worker: ' + response['error']
            if response['type'] == 'ValueError': raise ValueError(message)
            raise RuntimeError(message)
        return response

    def generate(self, prompt, sampling, options, seed):
        if self.closed: raise RuntimeError('Inference worker already closed')
        try:
            self.connection.send({'operation': 'generate', 'prompt': prompt, 'sampling': sampling, 'options': options, 'seed': seed})
        except (BrokenPipeError, EOFError, OSError) as error:
            raise RuntimeError('Inference worker exited unexpectedly; inspect its run log') from error
        response = self._receive()
        if response['kind'] != 'output' or not isinstance(response['text'], str):
            raise RuntimeError('Inference worker returned an invalid output')
        return response['text']

    def close(self):
        if self.closed: return
        self.closed = True
        if self.process.pid is not None:
            if self.process.is_alive():
                try: self.connection.send({'operation': 'close'})
                except (BrokenPipeError, EOFError, OSError): pass
            self.process.join(timeout=15)
            if self.process.is_alive(): self.process.terminate(); self.process.join(timeout=5)
            if self.process.is_alive(): self.process.kill(); self.process.join(timeout=5)
            if self.process.is_alive(): raise RuntimeError('Inference worker did not exit; refuse to load another model')
        self.connection.close()
        if self.cache:
            path = Path(self.cache)
            if path.parent == Path(tempfile.gettempdir()) and path.name.startswith('gecc-augmentation-model-'):
                shutil.rmtree(path, ignore_errors=True)
        if self.process.pid is not None:
            print(json.dumps({'augmentation_worker_exited': self.process.pid, 'exitcode': self.process.exitcode,
                              'memory_after_worker_exit': gpu_memory(self.device)}), flush=True)
