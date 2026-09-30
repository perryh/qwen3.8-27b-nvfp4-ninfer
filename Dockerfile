# syntax=docker/dockerfile:1
# Two-stage build of NInfer from a staged engine checkout (mirrors
# https://github.com/Neroued/ninfer Dockerfile: CUDA 13.1 devel -> runtime).
#
# run.sh stages the variant's engine source (content only, no .git) into the
# build context; SRC_DIR selects which staged tree to copy so the default and
# OrcaRouter variants can coexist in one working tree.
ARG SRC_DIR=ninfer-src

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS build

ARG SRC_DIR
ARG CMAKE_EXTRA_ARGS
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY ${SRC_DIR}/ .

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
        ${CMAKE_EXTRA_ARGS:-} \
    && cmake --build /build --parallel --target ninfer ninfer-serve

FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
