NVCC      = /usr/local/cuda-12.2/bin/nvcc
NVCCFLAGS = -O2 -ccbin gcc-12 -arch=compute_61 -code=sm_61 \
            -rdc=true -lcudadevrt \
            -DCUDA_FORCE_CDP1_IF_SUPPORTED \
            -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING
TARGET    = quicksort_cuda_dp
SRC       = quicksort_cuda_dp.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
