NVCC      = /usr/local/cuda-12.2/bin/nvcc
NVCCFLAGS = -O2 -ccbin gcc-12 -arch=compute_61 -code=sm_61
TARGET    = bfs_cuda
SRC       = bfs_cuda.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
