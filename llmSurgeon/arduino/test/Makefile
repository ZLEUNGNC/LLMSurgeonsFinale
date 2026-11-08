# Host-side tests for the panel's wire-format parser.
#
#   make          build and run
#   make clean
#
# Compiles ../panel/protocol.h -- the same header the sketch includes -- with a
# PC compiler, so the parser that ships is the parser that is tested.

CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -Wno-unused-parameter -O1

all: run

test_protocol: test_protocol.cpp ../panel/protocol.h
	$(CXX) $(CXXFLAGS) -o $@ test_protocol.cpp

run: test_protocol
	./test_protocol

clean:
	rm -f test_protocol test_protocol.exe

.PHONY: all run clean
