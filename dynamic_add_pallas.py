#!/usr/bin/env python3

"""dynamic_add_pallas.py — Dynamic-shape 2D add via a Pallas kernel.

Same as dynamic_add.py (shape polymorphism per
https://docs.jax.dev/en/latest/export/shape_poly.html), but the add is
a Pallas TPU kernel instead of plain jnp. Inputs are (n, 128): the lane
dimension is static at 128 (TPU vector geometry), only the sublane
dimension `n` is symbolic. The function is exported ONCE, then called
with several concrete sizes.

The polymorphic StableHLO (with the embedded Mosaic custom call) is
written to vector_add_pallas_dynamic.mlir.
"""

import os

DUMP_ROOT = "compiler_dump/"

HLO_DUMP_PATH = os.path.join(DUMP_ROOT, "hlo")

LLO_DUMP_PATH = os.path.join(DUMP_ROOT, "llo")

MOSAIC_DUMP_PATH = os.path.join(DUMP_ROOT, "mosaic")

os.makedirs(HLO_DUMP_PATH, exist_ok=True)

os.makedirs(LLO_DUMP_PATH, exist_ok=True)
os.makedirs(MOSAIC_DUMP_PATH, exist_ok=True)

# XLA flags — dump HLO
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_hlo_as_text "
    f"--xla_dump_to={HLO_DUMP_PATH} "
)

# libtpu flags — dump LLO
os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={LLO_DUMP_PATH} "
    f"--xla_jf_dump_hlo_text=true "
    f"--xla_jf_dump_llo_text=true "
    f"--xla_jf_emit_annotations=true "
    f"--xla_jf_debug_level=2 "
    f"--xla_mosaic_dump_to={MOSAIC_DUMP_PATH} "
)

# Import JAX AFTER setting env vars

import jax
import jax.numpy as jnp
from jax import export
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

def add(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )(x, y)

# Export once with a symbolic dimension `n` — the exported artifact is
# valid for any number of rows. Lowering targets TPU explicitly so the
# Mosaic kernel is generated even when exporting from a non-TPU host.
# Dynamically-shaped Pallas blocks are gated behind an experimental flag.
n, = export.symbolic_shape("n")

spec = jax.ShapeDtypeStruct((n, 128), jnp.float32)

with pl.pallas_export_experimental(dynamic_shapes=True):
    exported: export.Exported = export.export(jax.jit(add), platforms=["tpu"])(spec, spec)

print("in_avals: ", exported.in_avals)

print("out_avals:", exported.out_avals)

stablehlo = exported.mlir_module()

with open("vector_add_pallas_dynamic.mlir", "w") as f:
    f.write(stablehlo)

print("\nShape-polymorphic StableHLO (written to vector_add_pallas_dynamic.mlir):\n")

print(stablehlo)

# Call the SAME exported artifact with different concrete row counts.
if jax.default_backend() == "tpu":
    for rows in (1, 8, 100, 4099):
        x = jnp.arange(rows * 128, dtype=jnp.float32).reshape(rows, 128)
        y = jnp.ones((rows, 128), dtype=jnp.float32)
        result = exported.call(x, y)
        result.block_until_ready()
        expected = x + y
        ok = bool(jnp.allclose(result, expected))
        print(f"shape=({rows:5d}, 128)  ok={ok}  result[0,:3]={result[0, :3]}")
        assert ok, f"mismatch at rows={rows}"

    print(f"\nDumps written to:")

    print(f"  HLO:    {HLO_DUMP_PATH}")

    print(f"  Mosaic: {MOSAIC_DUMP_PATH}")

    print(f"  LLO:    {LLO_DUMP_PATH}")
else:
    print(f"\nbackend={jax.default_backend()} — skipping execution; run on a TPU VM to verify.")
