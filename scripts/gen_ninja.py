"""Write torch.utils.cpp_extension build files for HIP kernel sources without building or importing them.

Mirrors what `load(name=..., sources=[x.hip], extra_include_paths=[dir/include])` does in AMD's AIG eval_tools
(compile.py / kernel_loader_template.py) up to the ninja file: copy the source, hipify it in place (text only),
then torch's own `_write_ninja_file_to_build_library`. The caller runs `ninja` in a sandbox (run_compile.sh).
Never call load(): it imports the built module, which runs untrusted static initialisers.

usage: gen_ninja.py KERNEL_DIR OUT_DIR [--extra-cflag FLAG ...]   (run with the ROCm torch venv, $ZF_RUNNER_VENV)
"""
import argparse, os, shutil, signal, sys
import torch.utils.cpp_extension as ce
from torch.utils.hipify import hipify_python

ap = argparse.ArgumentParser()
ap.add_argument("kernel_dir"); ap.add_argument("out_dir")
ap.add_argument("--extra-cflag", action="append", default=[], help="added to both host and device flags")
ap.add_argument("--hipify-timeout", type=int, default=30, help="seconds per source; normal sources take < 1 s")
a = ap.parse_args()


def _hipify_timeout(signum, frame):  # hipify's launch parser is pure Python, so the alarm interrupts it
    raise TimeoutError(f"hipify took > {a.hipify_timeout} s")


# A malformed launch (`<<<...>>(`) sends hipify's parser into an endless loop whose memory grows until the OOM
# killer stops the whole process (220 GB RAM); bound each source with an alarm and skip it on timeout.
signal.signal(signal.SIGALRM, _hipify_timeout)
assert ce.IS_HIP_EXTENSION, "needs a ROCm build of torch"
# load() adds these (-L torch/lib -lc10 -ltorch ... -lamdhip64); without them the .so has undefined torch symbols
LDFLAGS = ce._prepare_ldflags([], True, False, False, False)
for f in sorted(os.listdir(a.kernel_dir)):
    if not f.endswith(".hip"):
        continue
    name = f[:-4]
    bdir = os.path.join(os.path.abspath(a.out_dir), name)
    src_dir = os.path.join(bdir, "hip")
    os.makedirs(os.path.join(src_dir, "include"), exist_ok=True)
    src = shutil.copy(os.path.join(a.kernel_dir, f), src_dir)
    incs = [os.path.join(src_dir, "include")]
    try:  # model-written sources can crash or hang hipify's parser (malformed <<<...>>> launches): skip them
        signal.alarm(a.hipify_timeout)
        res = hipify_python.hipify(project_directory=bdir, output_directory=bdir, header_include_dirs=incs,
                                   extra_files=[os.path.abspath(src)],
                                   ignores=[ce._join_rocm_home("*"), os.path.join(ce._TORCH_PATH, "*")],
                                   show_detailed=False, show_progress=False, is_pytorch_extension=True)
    except Exception as e:  # no build.ninja -> the compile step builds no .so -> graded "compiled": false
        print(f"hipify failed for {f}: {type(e).__name__}: {e}", file=sys.stderr)
        continue
    finally:
        signal.alarm(0)
    s = res[src].hipified_path if src in res and res[src].hipified_path else src
    ce._write_ninja_file_to_build_library(
        path=os.path.join(bdir, "build.ninja"), name=name, sources=[s], extra_cflags=list(a.extra_cflag),
        extra_cuda_cflags=list(a.extra_cflag), extra_sycl_cflags=[], extra_ldflags=LDFLAGS, extra_include_paths=incs,
        with_cuda=True, with_sycl=False, is_standalone=False)
print("wrote", len([d for d in os.listdir(a.out_dir)]), "build dirs in", a.out_dir, file=sys.stderr)
