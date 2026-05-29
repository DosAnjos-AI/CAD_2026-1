CC      = gcc
CFLAGS  = -O2 -fopenmp
TARGET  = mergesort_omp
SRC     = mergesort_omp.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
