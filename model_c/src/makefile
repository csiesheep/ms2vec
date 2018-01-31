SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

all: ms2vec

ms2vec: ms2vec.c
	$(CC) ms2vec.c -o ${BIN_DIR}/ms2vec $(CFLAGS)

clean:
	pushd ${BIN_DIR} && rm -rf ms2vec; popd
