"""Select one warmed-up eager PyTorch softmax launch for Nsight Compute."""
import torch

@torch.no_grad()
def main():
    x = ((torch.arange(4096 * 1024, device='cuda') % 41 - 20) * 0.25).float().reshape(4096, 1024)
    y = torch.empty_like(x)
    ref = torch.softmax(x.double(), dim=-1)
    for _ in range(50):
        torch.softmax(x, dim=-1, out=y)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.softmax(x, dim=-1, out=y)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    torch.testing.assert_close(y.double(), ref, atol=1e-7, rtol=1e-5)
    assert torch.isfinite(y).all().item()
    assert ((y >= 0) & (y <= 1)).all().item()
    assert ((y.double().sum(dim=-1) - 1).abs() <= 1e-5).all().item()
    print('PASS: PyTorch profiled output matches FP64 reference and probability invariants.')

if __name__ == '__main__':
    main()
