CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -g
CC ?= gcc
CFLAGS_APP ?= -O2 -no-pie -fno-pie -Iapplications
B := build

all: $(B)/cachesim $(B)/multisim $(B)/lackey2trace $(B)/labeler $(B)/winexport

$(B)/cachesim: simulator/cachesim.cpp simulator/*.h | $(B)
	$(CXX) $(CXXFLAGS) -o $@ simulator/cachesim.cpp

$(B)/multisim: simulator/multisim.cpp simulator/*.h | $(B)
	$(CXX) $(CXXFLAGS) -o $@ simulator/multisim.cpp

$(B)/labeler: belady/labeler.cpp belady/labeler.h simulator/*.h | $(B)
	$(CXX) $(CXXFLAGS) -o $@ belady/labeler.cpp

$(B)/winexport: export/winexport.cpp simulator/*.h | $(B)
	$(CXX) $(CXXFLAGS) -o $@ export/winexport.cpp

$(B)/lackey2trace: trace_collection/lackey2trace.cpp trace_collection/lackey_parser.h simulator/trace.h | $(B)
	$(CXX) $(CXXFLAGS) -o $@ trace_collection/lackey2trace.cpp

$(B)/run_multi_tests: tests/test_multi.cpp simulator/*.h | $(B)
	$(CXX) $(CXXFLAGS) -Isimulator -o $@ tests/test_multi.cpp

$(B)/run_tests: tests/test_main.cpp simulator/*.h belady/labeler.h trace_collection/lackey_parser.h | $(B)
	$(CXX) $(CXXFLAGS) -Isimulator -Itrace_collection -o $@ tests/test_main.cpp

$(B):
	mkdir -p $(B)

# Workloads must be non-PIE so the address of roi_marker is fixed and can be read with nm.
APPS := matrix graph conv hash
workloads: $(addprefix $(B)/apps/,$(APPS)) $(B)/apps/seq
$(B)/apps/%: applications/%.c applications/common.h | $(B)/apps
	$(CC) $(CFLAGS_APP) -o $@ $<
$(B)/apps/seq: tests/programs/seq.c applications/common.h | $(B)/apps
	$(CC) $(CFLAGS_APP) -o $@ $<
$(B)/apps:
	mkdir -p $(B)/apps

test: all $(B)/run_tests $(B)/run_multi_tests workloads
	$(B)/run_tests
	$(B)/run_multi_tests
	python3 tests/test_dataset.py
	python3 tests/test_models.py
	python3 tests/test_lstm.py
	python3 tests/test_dagger.py
	python3 tests/test_multi.py
	python3 tests/test_mixed.py
	python3 tests/test_mixed_pipeline.py
	python3 tests/test_final.py
	python3 tests/test_winexport.py
	python3 tests/test_make_windows.py
	bash tests/e2e_seq.sh

clean:
	rm -rf $(B)

.PHONY: all test clean workloads
