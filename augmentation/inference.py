"""Native parameter audits and sequential FP16 generation, with no model training."""
import gc
import math
import tempfile


def layer_placement(model, device):
    import torch
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > 8_000_000_000:
        raise ValueError(f'Native model exceeds8B: {parameters}')
    if device == 'cpu':
        return {'parameters': parameters, 'device_map': {'': 'cpu'}, 'memory_budget': {}, 'tied_embeddings':
                model.get_input_embeddings().weight is model.get_output_embeddings().weight}
    if device != 'cuda' or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Committed augmentation requires GPU T4 x2; no offload')
    decoder = model.model
    tied = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    sizes = [sum(p.numel() * p.element_size() for p in layer.parameters()) for layer in decoder.layers]
    embedding = sum(p.numel() * p.element_size() for p in decoder.embed_tokens.parameters())
    ending = sum(p.numel() * p.element_size() for p in decoder.norm.parameters())
    if not tied: ending += sum(p.numel() * p.element_size() for p in model.lm_head.parameters())
    split = min(range(1, len(sizes)), key=lambda i: abs(embedding + sum(sizes[:i]) - ending - sum(sizes[i:])))
    mapping = {'model.embed_tokens': 'cuda:0', 'model.norm': 'cuda:1', 'lm_head': 'cuda:0' if tied else 'cuda:1'}
    if hasattr(decoder, 'rotary_emb'): mapping['model.rotary_emb'] = 'cuda:0'
    mapping.update({f'model.layers.{i}': f'cuda:{int(i >= split)}' for i in range(len(sizes))})
    weights = {}
    for name, p in model.named_parameters():
        matches = [key for key in mapping if name == key or name.startswith(key + '.')]
        if not matches: raise ValueError(f'Explicit device map does not cover {name}')
        destination = mapping[max(matches, key=len)]
        weights[destination] = weights.get(destination, 0) + p.numel() * p.element_size()
    budgets = {}
    for destination, size in weights.items():
        free, total = torch.cuda.mem_get_info(int(destination.split(':')[1]))
        required = math.ceil(1.10 * size + 1.5 * 1024**3)
        if required > free:
            raise RuntimeError(f'{destination} needs estimated {required / 1024**3:.2f} GiB; free {free / 1024**3:.2f} GiB')
        budgets[destination] = {'weight_bytes': size, 'required_bytes': required, 'free_bytes': free, 'total_bytes': total}
    return {'parameters': parameters, 'device_map': mapping, 'memory_budget': budgets, 'tied_embeddings': tied}


def native_audit(settings, device, cache_dir=None):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(settings['name'], revision=settings['revision'], cache_dir=cache_dir)
    if getattr(config, 'quantization_config', None) is not None:
        raise ValueError('Use a native unquantized checkpoint for augmentation')
    AutoModelForCausalLM._model_mapping[type(config)]
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16 if device == 'cuda' else torch.float32)
    model.tie_weights()
    placement = layer_placement(model, device)
    if placement['parameters'] != settings['parameters']:
        raise ValueError(f'Native parameter count differs from pinned model audit: {placement["parameters"]}')
    return {**placement, 'model': settings, 'resolved_revision': config._commit_hash or settings['revision'],
            'compute_dtype': 'float16' if device == 'cuda' else 'float32', 'quantized': False}


class NativeEngine:
    """One native model at a time; its download cache never enters dataset outputs."""
    def __init__(self, settings, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cache = tempfile.TemporaryDirectory(prefix='gecc-augmentation-model-')
        self.model = self.tokenizer = self.constraint_tokens = None
        try:
            self.audit = native_audit(settings, device, self.cache.name)
            self.tokenizer = AutoTokenizer.from_pretrained(settings['name'], revision=settings['revision'], cache_dir=self.cache.name)
            self.model = AutoModelForCausalLM.from_pretrained(settings['name'], revision=settings['revision'],
                dtype=torch.float16 if device == 'cuda' else torch.float32,
                device_map=self.audit['device_map'], low_cpu_mem_usage=True,
                attn_implementation='sdpa', cache_dir=self.cache.name)
            self.model.tie_weights()
            if sum(p.numel() for p in self.model.parameters()) != self.audit['parameters']:
                raise RuntimeError('Loaded parameter count differs from audited unique native count')
            expected = set(self.audit['device_map'].values())
            if any(str(p.device) not in expected for p in self.model.parameters()):
                raise RuntimeError('Loaded weights differ from explicit map; offload forbidden')
            self.model.eval()
            self.device = self.model.get_input_embeddings().weight.device
            marker = 'GECC_INPUT_CONTENT_SENTINEL'
            self.role = None
            for role in ('user', 'question'):
                try:
                    rendered = self.tokenizer.apply_chat_template([{'role': role, 'content': marker}], tokenize=False, add_generation_prompt=True)
                    if marker in rendered: self.role = role; break
                except (ValueError, KeyError): pass
            if self.role is None: raise ValueError('Chat template drops the user input for both supported roles')
        except BaseException:
            self.close(); raise

    def generate(self, prompt, sampling, options, seed):
        import torch
        from transformers import set_seed
        set_seed(seed)
        rendered = self.tokenizer.apply_chat_template([{'role': self.role, 'content': prompt}], tokenize=False, add_generation_prompt=True)
        batch = self.tokenizer(rendered, return_tensors='pt').to(self.device)
        if batch.input_ids.shape[1] > options['max_input_tokens']:
            raise ValueError('Prompt exceeds the committed token budget; never truncate source evidence')
        generation = {'max_new_tokens': options['max_new_tokens'], 'do_sample': sampling,
                      'pad_token_id': self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                      'use_cache': True, 'max_time': options.get('max_time', 180)}
        if options.get('response_schema') is not None:
            from .constraints import tokenizer_data, prefix_filter
            if self.constraint_tokens is None:
                self.constraint_tokens = tokenizer_data(self.tokenizer)
            generation['prefix_allowed_tokens_fn'] = prefix_filter(self.constraint_tokens, options['response_schema'])
        if sampling: generation.update(temperature=options['temperature'], top_p=options['top_p'])
        with torch.inference_mode(): outputs = self.model.generate(**batch, **generation)
        return self.tokenizer.decode(outputs[0][batch.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def close(self):
        import torch
        self.model = self.tokenizer = self.constraint_tokens = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if getattr(self, 'cache', None): self.cache.cleanup()
