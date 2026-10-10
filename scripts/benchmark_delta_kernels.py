"""Isolated Qwen3.5 delta-rule probe; does not load or alter model weights.

Run with FLA available only on this process's PYTHONPATH. Checks finite outputs,
forward agreement, gradients, and synchronized warm timing on representative
shapes. Microbenchmark speedups do not establish end-to-end training speedups.
"""
import inspect
import importlib.metadata
import json
from pathlib import Path
import time

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule


torch.manual_seed(4242)
reference_chunk = inspect.unwrap(torch_chunk_gated_delta_rule)
reference_recurrent = inspect.unwrap(torch_recurrent_gated_delta_rule)
results = {}
for name, length, reference, optimized in [
    ('chunk_forward_backward', 256, reference_chunk, chunk_gated_delta_rule),
    ('recurrent_forward', 1, reference_recurrent, fused_recurrent_gated_delta_rule),
]:
    shape = (1, length, 16, 128)
    inputs = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    inputs += [-torch.rand((1, length, 16), device='cuda'), torch.rand((1, length, 16), device='cuda', dtype=torch.bfloat16)]
    kwargs = {'use_qk_l2norm_in_kernel': True}
    if length == 1:
        kwargs.update(initial_state=torch.randn((1, 16, 128, 128), device='cuda'), output_final_state=True)
    outputs, gradients, timings = {}, {}, {}
    for implementation, function in [('reference', reference), ('fla', optimized)]:
        tensors = [x.detach().clone().requires_grad_(length > 1) for x in inputs]
        def invoke():
            for tensor in tensors:
                tensor.grad = None
            output, state = function(*tensors, **kwargs)
            if length > 1:
                output.float().square().mean().backward()
            return output, state
        output, state = invoke()  # Compilation/warmup excluded from timing.
        outputs[implementation] = output.detach().float()
        if not torch.isfinite(output).all() or (state is not None and not torch.isfinite(state).all()):
            raise ValueError(f'{name}/{implementation}: nonfinite forward result')
        if length > 1:
            gradients[implementation] = [x.grad.detach().float().clone() for x in tensors]
            if not all(torch.isfinite(g).all() for g in gradients[implementation]):
                raise ValueError(f'{name}/{implementation}: nonfinite gradient')
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(3):
            invoke()
        torch.cuda.synchronize()
        timings[implementation] = (time.perf_counter() - started) / 3
    difference = outputs['reference'] - outputs['fla']
    relative_l2 = (difference.norm() / outputs['reference'].norm().clamp_min(1e-12)).item()
    # BF16 implementations use different accumulation order. This probe reports
    # numerical differences explicitly; it does not promise bitwise equivalence.
    if relative_l2 > 0.02:
        raise ValueError(f'{name}: relative output error {relative_l2} exceeds 2% probe threshold')
    row = {'shape': shape, 'seconds': timings, 'relative_output_l2_error': relative_l2,
           'max_absolute_output_error': difference.abs().max().item(),
           'warm_speedup': timings['reference'] / timings['fla']}
    if gradients:
        row['relative_gradient_l2_errors'] = [((a-b).norm()/a.norm().clamp_min(1e-12)).item()
                                             for a,b in zip(gradients['reference'], gradients['fla'], strict=True)]
        if max(row['relative_gradient_l2_errors']) > 0.05:
            raise ValueError(f'{name}: relative gradient error exceeds 5% probe threshold')
    results[name] = row
    print(json.dumps({name: row}), flush=True)
report = {'gpu': torch.cuda.get_device_name(), 'packages': {p: importlib.metadata.version(p) for p in
          ('torch', 'triton', 'transformers', 'flash-linear-attention', 'fla-core', 'einops')},
          'results': results, 'scope': 'Isolated kernel probe. Concurrent GPU work may affect timings; requires a separate end-to-end training check before adoption.'}
Path('/content/kernel-probe-result.json').write_text(json.dumps(report, indent=2)+'\n')
