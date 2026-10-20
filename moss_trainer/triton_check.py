"""Run in the selected training Python, never in the GUI environment."""
import json
import sys


def main():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add_one(source, target, count: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(source + offsets, offsets < count, other=0)
        tl.store(target + offsets, values + 1, offsets < count)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable')
    source = torch.arange(257, dtype=torch.float32, device='cuda')
    target = torch.empty_like(source)
    add_one[(triton.cdiv(source.numel(), 128),)](source, target, source.numel(), BLOCK=128)
    torch.cuda.synchronize()
    torch.testing.assert_close(target, source + 1)
    print(json.dumps({'triton_runtime': True, 'triton_version': triton.__version__,
                      'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'triton_runtime': False, 'error': str(exc)}))
        sys.exit(1)
