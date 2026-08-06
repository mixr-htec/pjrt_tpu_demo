#!/usr/bin/env python3

"""dynamic_add.py — Dynamic-shape vector add via JAX shape polymorphism.

Exports x + y ONCE with a symbolic length `n` (see
https://docs.jax.dev/en/latest/export/shape_poly.html), producing a
shape-polymorphic StableHLO module with tensor<?xf32> operands, then
calls it with several concrete sizes and verifies the results.

The polymorphic StableHLO is written to vector_add_dynamic.mlir.
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

def add(x, y):
    return x + y

# Export once with a symbolic dimension `n` — the exported artifact is
# valid for any vector length.
n, = export.symbolic_shape("n")

spec = jax.ShapeDtypeStruct((n,), jnp.float32)

exported: export.Exported = export.export(jax.jit(add))(spec, spec)

print("in_avals: ", exported.in_avals)

print("out_avals:", exported.out_avals)

stablehlo = exported.mlir_module()

with open("vector_add_dynamic.mlir", "w") as f:
    f.write(stablehlo)

print("\nShape-polymorphic StableHLO (written to vector_add_dynamic.mlir):\n")

print(stablehlo)

# Call the SAME exported artifact with different concrete sizes.
for size in (8, 100, 4099, 8192):
    x = jnp.arange(size, dtype=jnp.float32)
    y = jnp.ones((size,), dtype=jnp.float32)
    result = exported.call(x, y)
    result.block_until_ready()
    expected = x + y
    ok = bool(jnp.allclose(result, expected))
    print(f"size={size:5d}  ok={ok}  result[:3]={result[:3]}")
    assert ok, f"mismatch at size={size}"

print(f"\nDumps written to:")

print(f"  HLO:    {HLO_DUMP_PATH}")

print(f"  Mosaic: {MOSAIC_DUMP_PATH}")

print(f"  LLO:    {LLO_DUMP_PATH}")
