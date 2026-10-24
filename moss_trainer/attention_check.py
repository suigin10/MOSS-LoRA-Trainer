"""Small forward/backward GPU checks for optional sd-scripts attention kernels."""
import argparse
import json


def check(backend):
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with bf16 support is required')
    if backend == 'xformers':
        from xformers.ops import memory_efficient_attention
        attention = lambda q, k, v: memory_efficient_attention(q, k, v, p=0.0)
    else:
        from flash_attn import flash_attn_func
        attention = lambda q, k, v: flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    torch.manual_seed(42)
    for length in (127, 256):
        inputs = [torch.randn(1, length, 8, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        output = attention(*inputs)
        expected = torch.nn.functional.scaled_dot_product_attention(
            *(x.detach().transpose(1, 2) for x in inputs), dropout_p=0.0).transpose(1, 2)
        torch.testing.assert_close(output, expected, atol=0.03, rtol=0.03)
        output.float().square().mean().backward()
        if not all(x.grad is not None and torch.isfinite(x.grad).all().item() for x in inputs):
            raise RuntimeError('Non-finite or missing attention gradients')
    torch.cuda.synchronize()
    print(json.dumps({'backend': backend, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(0), 'forward_backward': 'ok'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('backend', choices=['xformers', 'flash'])
    check(parser.parse_args().backend)
