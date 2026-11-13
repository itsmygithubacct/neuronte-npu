# Build the NPU backend and benchmark on the target board.
#
#   make NEURONTE_DIR=/path/to/neuronte RKNN_INCLUDE=/path/to/rknn/include
#
# NEURONTE_DIR must contain include/neuronte.h and build/libneuronte.a (the
# reference CPU library, used for the CPU path and the fallback).
# RKNN_INCLUDE must contain rknn_api.h from the RKNN SDK. The RKNN runtime
# library is loaded at run time with dlopen, so the binary also runs, on the
# CPU path, on boards with no NPU or no runtime installed.

CC ?= cc
CFLAGS ?= -O3
NEURONTE_DIR ?= ../neuronte
RKNN_INCLUDE ?= /usr/include
WARN = -Wall -Wextra -Wno-unused-parameter

CPPFLAGS += -Isrc -I$(NEURONTE_DIR)/include -I$(RKNN_INCLUDE)
LDLIBS += $(NEURONTE_DIR)/build/libneuronte.a -ldl -lm -lpthread

all: build/neuronte_npu_bench build/rknn_probe

build:
	mkdir -p build

build/neuronte_npu.o: src/neuronte_npu.c src/neuronte_npu.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -c -o $@ $<

build/neuronte_npu_bench: bench/neuronte_npu_bench.c build/neuronte_npu.o | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -o $@ $^ $(LDLIBS)

build/rknn_probe: bench/rknn_probe.c | build
	$(CC) -I$(RKNN_INCLUDE) $(CFLAGS) $(WARN) -o $@ $< -ldl -lm

clean:
	rm -rf build

.PHONY: all clean
