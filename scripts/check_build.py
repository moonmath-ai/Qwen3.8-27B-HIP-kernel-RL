"""The model card's build snippet, run as written: build kernel.hip with torch.utils.cpp_extension.load and check
it against the task's functional Model. Args: kernel.hip functional.py"""
import importlib.util, sys, torch
from torch.utils.cpp_extension import load
hip, fpy = sys.argv[1], sys.argv[2]
ext = load(name="cand", sources=[hip], extra_cflags=["-O3"], extra_cuda_cflags=["-O3"])
spec = importlib.util.spec_from_file_location("task", fpy); task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)
torch.manual_seed(0)
model = task.Model(*task.get_init_inputs()).cuda()
inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in task.get_inputs()]
with torch.no_grad():
    ref = model(*inputs)                   # module_fn, the PyTorch reference
    out = model(*inputs, fn=ext.forward)   # the same Model calling the compiled kernel
torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
print("BUILD_CHECK_OK", torch.__version__)
