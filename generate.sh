#!/bin/sh

builder="-G Ninja"

rapidjson_root='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/tianrunhe/packages/rapidjson-release'
rapidjson_inc=${rapidjson_root}/include

if [ "$1" == "make" ]; then
    builder=""
fi

RapidJSON_DIR=${rapidjson_root} \
cmake ${builder} .. \
    -DCMAKE_CXX_FLAGS=-I${rapidjson_inc} \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DBUILD_PY_FFI=ON \
    -DBUILD_MULTI_GPU=ON \
    -DCMAKE_CUDA_FLAGS="-lineinfo" \
    -DUSE_NVTX=ON
