# Fixed-physical-depth FVCOM currents

RUTide's fixed-depth vector mode analyzes FVCOM `u(time, siglay, nele)` and
`v(time, siglay, nele)` after interpolating the native terrain-following layer
centres to requested physical depths. This document freezes the vertical and
missing-value semantics used by the command-line application.

## Public convention

`--depths D1,D2,...` accepts finite, unique values strictly greater than zero.
Each value is metres below the **instantaneous free surface**; positive is
downward. `--depths` is mutually exclusive with `--layers` and `--layer-count`.
Requested order is retained in the output.

The required FVCOM inputs are:

- `u(time, siglay, nele)` and `v(time, siglay, nele)`;
- `siglay(siglay, node)`, `h(node)`, and `zeta(time, node)`;
- one-based triangular connectivity `nv(three, nele)`; and
- `wet_cells(time, nele)` as the authoritative wet/dry mask.

The layer-centre depth below the instantaneous surface at an element centroid
is the arithmetic mean of the three nodal depths

```text
d(k, e, t) = mean_i[-siglay(k, node_i) * (h(node_i) + zeta(t, node_i))]
```

for the three nodes in element `e`. This is the centroid value of a field that
is linear over an FVCOM triangle and remains correct when sigma coordinates
vary horizontally. It deliberately does not multiply independently averaged
sigma and water depth.

## Interpolation and missing data

For each time, element, and target depth, RUTide locates the two adjacent
physical layer centres and linearly interpolates both current components using
the same weight. Exact layer-centre matches use that layer directly.

RUTide does not extrapolate above the shallowest layer centre or below the
deepest layer centre. A target sample is jointly missing when any of these
conditions holds:

- `wet_cells` is zero or missing;
- a required nodal `siglay`, `h`, or `zeta` value is missing;
- physical layer-centre depths are not finite and strictly increasing;
- the target is outside the layer-centre bracket; or
- either bracketing `u`/`v` component is missing.

The harmonic solver receives the resulting joint finite mask. Because a
rectangular depth-by-element product legitimately includes shallow or always-dry
coordinates, a series without enough observations for its requested model is
retained with `analysis_status=unavailable` and NaN-valued harmonic results. It
is never silently treated as a successful fit. NetCDF uses flag value `0` for
`fitted` and `1` for `unavailable`; JSON reports count both classes.

## Storage and performance contract

Coefficient and diagnostic variables use `(depth, element[, constituent])`;
reconstruction uses `(time, depth, element)`. `depth` has units `m`,
`positive="down"`, and the instantaneous-free-surface reference is explicit in
global metadata. `element_index` preserves the requested source order.

Input is processed in bounded element blocks. A contiguous block reads each
complete `u` and `v` time-layer-element hyperslab with one NetCDF request,
retains native `f32` storage until values enter the `f64` interpolation kernel,
and computes time rows in parallel. Nodal `zeta` uses a bounded contiguous span
when the span is no more than six times the selected-node count; otherwise it
falls back to sparse gathers. Wet cells are compacted to a byte mask. Sparse
element selections retain the general gather path.

Each block reads every native source layer once, computes all requested depths,
and then solves each target slice. Consequently, requesting additional depths
increases interpolation, solver, and output work but does not reread the
expensive native `u`/`v` block. The automatic memory planner conservatively
accounts for the bounded zeta span and typed source buffers. Results use the
same incremental, transactional NetCDF path as native sigma layers.
