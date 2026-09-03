# Pre-filter transfer-function correction

Pre-filter correction is the one scientifically useful MATLAB UTide analysis
option that RUTide has deliberately not yet implemented. It is optional for raw
FVCOM fields and most raw ADCP workflows; it matters when observations were
filtered before harmonic analysis and the filter appreciably attenuates a
selected tidal frequency.

## Upstream behavior

MATLAB UTide accepts frequency/gain pairs in cycles per hour, linearly
interpolates one gain per fitted constituent, replaces an out-of-range or
unacceptable interpolated gain with unity, and multiplies the astronomical
basis by that gain during solve and reconstruction. The retained Python port
still passes an always-empty legacy value through solve/reconstruction, but its
harmonic implementation labels the option unimplemented and comments out the
MATLAB interpolation.

This means the feature is a MATLAB restoration, not Python-UTide parity.

## Value and limits

If a known preprocessing filter has frequency response `P(f)`, incorporating
that response in the design matrix can estimate pre-filter harmonic amplitudes
without pretending that attenuated observations were raw. It does not recover
information removed by a stop-band, correct an unknown processing chain, or
repair aliasing. Near-zero response is intrinsically ill-conditioned and must
be rejected rather than amplified into a plausible-looking result.

Raw FVCOM `zeta`, `ua`/`va`, and native `u`/`v` should continue to use the unity
response. The feature is mainly useful for filtered instrument records and
model products whose preprocessing response is known and documented.

## RUTide implementation contract

The implementation should be a typed, opt-in frequency-response object rather
than an unvalidated collection of arrays. It must:

1. validate finite, strictly increasing frequency samples and finite gains;
2. define explicit interpolation and outside-range behavior;
3. reject gains too close to zero with a constituent-specific error;
4. precompute the response once per selected constituent, outside sample and
   series loops;
5. apply the same response to ordinary, exact/approximate inferred, scalar, and
   vector design bases;
6. retain enough response metadata to reconstruct in the fitted observation
   domain and to make any de-filtered product explicit;
7. serialize configuration and resolved constituent gains in Python snapshots
   and FVCOM output provenance; and
8. test synthetic known-filter recovery, reconstruction, inference, invalid
   responses, and the unity fast path against the MATLAB equations.

The unity path must remain allocation-free in the inner kernel. With a valid
response, runtime overhead should be negligible because the selected-frequency
gains are computed once and folded into existing cached bases.

## Primary references

- MATLAB UTide `ut_solv.m`, revision `4a6354f`, option documentation and
  `ut_E` implementation.
- Python UTide `harmonics.py`, revision `9b60caf`, where the pre-filter argument
  is explicitly marked unimplemented and the MATLAB interpolation is commented
  out.
