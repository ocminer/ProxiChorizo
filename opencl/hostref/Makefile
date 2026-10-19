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
selftest-%: selftest.c shim.h ../proxichorizo.cl
	$(CC) $(CFLAGS) $(DEFS) -DPX_FE=$* -I.. -o $@ selftest.c
check: all
	@for f in $(FES); do ./fetest-$$f || exit 1; ./selftest-$$f ../../vectors.json || exit 1; done
clean:
	rm -f $(BINS)
.PHONY: all check clean
