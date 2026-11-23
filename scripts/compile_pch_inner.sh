#!/bin/bash
# Inside sandbox.sh: build every gen_ninja.py build dir under ROOT with a per-side precompiled header.
# 1) host + gfx942 device PCH of <hip/hip_runtime.h> + <torch/extension.h>, built once from the first dir's flags
#    minus -DTORCH_EXTENSION_NAME (differs per kernel). The device PCH runs the driver's cc1 job with -emit-pch,
#    because hipcc --cuda-device-only -c insists on linking.
# 2) per kernel: the driver's job list (hipcc ... -###) with -include-pch added to each cc1 job, then the same
#    link as the ninja file ($cxx obj -shared -o name.so). Prints "name rc ms"; "# pch_build_ms N" first.
# usage: compile_pch_inner.sh ROOT J
set -u
ROOT=$1; J=$2; P=$ROOT/_pch; mkdir -p $P; HIPCC=${ZF_ROCM:-/opt/rocm-7.2.4}/bin/hipcc
first=$(ls -d $ROOT/*/build.ninja | head -1)
FL=$(sed -n 's/^cuda_cflags = //p' $first | sed 's/-DTORCH_EXTENSION_NAME=[^ ]*//')
printf '#pragma once\n#include <hip/hip_runtime.h>\n#include <torch/extension.h>\n' > $P/pch.h
s=$(date +%s%N)
$HIPCC $FL --cuda-host-only -c -x hip -Xclang -emit-pch $P/pch.h -o $P/host.pch > $P/host.log 2>&1 &
$HIPCC $FL --cuda-device-only -c -x hip $P/pch.h -o $P/x.o -### 2>&1 | python3 -c "
import shlex,sys
for l in sys.stdin:
    if l.strip().startswith('\"') and '\"-cc1\"' in l:
        a=shlex.split(l.strip()); a=['-emit-pch' if x=='-emit-obj' else x for x in a]; a[a.index('-o')+1]='$P/dev.pch'; print(shlex.join(a)); break
" > $P/devpch.sh
bash $P/devpch.sh > $P/dev.log 2>&1; wait
echo "# pch_build_ms $(( ($(date +%s%N)-s)/1000000 )) host=$(stat -c %s $P/host.pch 2>/dev/null) dev=$(stat -c %s $P/dev.pch 2>/dev/null)"
one() {
  d=$1; n=$(basename $d); s=$(date +%s%N)
  [ -f $d/$n.so ] && { echo "$n 0 0 cached"; return; }  # names are content hashes: an existing .so is this source
  FL=$(sed -n 's/^cuda_cflags = //p' $d/build.ninja); SRC=$(sed -n 's/^build .*cuda_compile \(.*\)$/\1/p' $d/build.ninja)
  OBJ=$d/$(sed -n 's/^build \(.*\.cuda\.o\): .*/\1/p' $d/build.ninja)
  { $HIPCC $FL -c $SRC -o $OBJ -### 2>&1 | python3 -c "
import shlex,sys
print('set -e')
for l in sys.stdin:
    l=l.strip()
    if not l.startswith('\"'): continue
    a=shlex.split(l)
    if '-cc1' in a: a[2:2]=['-include-pch','$P/'+('dev.pch' if '-fcuda-is-device' in a else 'host.pch')]
    print(shlex.join(a))
" > $d/jobs_pch.sh && bash $d/jobs_pch.sh && c++ $OBJ $(sed -n 's/^ldflags = //p' $d/build.ninja) -o $d/$n.so; } > $d/build.log 2>&1
  echo "$n $? $(( ($(date +%s%N)-s)/1000000 ))"
}
export -f one; export HIPCC P
ls -d $ROOT/*/ | grep -v '/_pch/$' | sed 's#/$##' | xargs -P $J -I{} bash -c 'one {}'
