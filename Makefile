CC = gcc
#CFLAGS = -fPIC -Wall -g
CFLAGS = -fPIC -O3
LDFLAGS = -shared
RM = rm -rf
TARGET_LIB = RobustPeakFinder.so

SRCS = RobustPeakFinder.c RGFLib.c
OBJS = $(SRCS:.c=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

.PHONY: test
test:
	python3 RobustPeakFinder_Python_Test_Simple.py

.PHONY: clean
clean:
	-${RM} ${TARGET_LIB} ${OBJS} $(SRCS:.c=.d)
