CC      = gcc
CFLAGS  = -O2 -fopenmp
TARGET  = quicksort_omp
SRC     = quicksort_omp.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
