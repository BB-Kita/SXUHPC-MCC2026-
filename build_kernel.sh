#!/bin/bash
# 编译 C++/OpenMP 融合核 -> clim_kernel.so
set -eo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

# devtoolset-7 提供 g++ (默认 PATH 里没有)
if [ -f /opt/rh/devtoolset-7/enable ]; then
    source /opt/rh/devtoolset-7/enable
fi
GXX=${GXX:-$(command -v g++ || echo /opt/rh/devtoolset-7/root/usr/bin/g++)}

echo "using: $GXX  ($($GXX --version | head -1))"
"$GXX" -O3 -fopenmp -march=native -funroll-loops -fPIC -shared -std=c++14 \
    -o clim_kernel.so clim_kernel.cpp

echo "built: $(pwd)/clim_kernel.so"
