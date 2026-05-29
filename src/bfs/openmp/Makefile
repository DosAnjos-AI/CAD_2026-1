CC      = gcc
CFLAGS  = -O2 -fopenmp
TARGET  = bfs_omp
SRC     = bfs_omp.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
