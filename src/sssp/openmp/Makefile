CC      = gcc
CFLAGS  = -O2 -fopenmp
TARGET  = sssp_omp
SRC     = sssp_omp.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
