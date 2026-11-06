# CPU reference build of proxichorizo.cl: the OpenCL source compiled as plain C with shim.h, so the
# field arithmetic and the whole VRF attempt can be run and debugged on the host. With PX_HOST_REF
# the fe29 layout asserts its limb invariants (operands < 2^30 before a multiply, subtrahend limbs
# below the 1024p constant), which a GPU would silently violate.
#   make            builds fetest-<fe> and selftest-<fe> for fe 29, 32, 51
#   make check      runs them (selftest needs ../../vectors.json)
FES ?= 29 32 51
CFLAGS ?= -O1 -std=gnu11 -Wall -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-variable
DEFS = -DPX_WINDOW=5 -DPX_COMPACT=1 -DPX_WG=256
BINS = $(foreach f,$(FES),fetest-$(f) selftest-$(f))
all: $(BINS)
fetest-%: fetest.c shim.h ../proxichorizo.cl
	$(CC) $(CFLAGS) $(DEFS) -DPX_FE=$* -I.. -o $@ fetest.c
selftest-%: selftest.c shim.h ../proxichorizo.cl ../merged_scalar.h ../job_midstate.h
	$(CC) $(CFLAGS) $(DEFS) -DPX_FE=$* -I.. -o $@ selftest.c
check: all
	@for f in $(FES); do ./fetest-$$f || exit 1; ./selftest-$$f ../../vectors.json || exit 1; done
ladder-edges: ladder_edges.c shim.h ../proxichorizo.cl ../merged_scalar.h
	$(CC) $(CFLAGS) $(DEFS) -DPX_FE=29 -DPX_EDWARDS_OPT=1 -DPX_LADDER=1 -DPX_XONLY=1 -DPX_JACOBI=1 -DPX_SHA_MID=1 -DPX_SPLIT_FINISH=16 -I.. -o $@ ladder_edges.c
clean:
	rm -f $(BINS) ladder-edges
.PHONY: all check clean
