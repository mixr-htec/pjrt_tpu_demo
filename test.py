#!/usr/bin/env python3

"""minimal_pallas_add.py — Print Mosaic IR + LLO for vector add on TPU."""

import os

DUMP_ROOT = "compiler_dump/"

HLO_DUMP_PATH = os.path.join(DUMP_ROOT, "hlo")

LLO_DUMP_PATH = os.path.join(DUMP_ROOT, "llo")

MOSAIC_DUMP_PATH = os.path.join(DUMP_ROOT, "mosaic")

os.makedirs(HLO_DUMP_PATH, exist_ok=True)

os.makedirs(LLO_DUMP_PATH, exist_ok=True)
os.makedirs(MOSAIC_DUMP_PATH, exist_ok=True)

# XLA flags — dump HLO + Mosaic MLIR passes
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
from jax.experimental import pallas as pl

def slice_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[0:2, 0:2, 8:11, 111:211]

x = jnp.arange(3 * 3 * 14 * 216, dtype=jnp.float32).reshape((3, 3, 14, 216))

result = pl.pallas_call(
    slice_kernel,
    out_shape=jax.ShapeDtypeStruct((2, 2, 3, 100), jnp.float32),
    debug=True,
)(x)

result.block_until_ready()

expected = x[0:2, 0:2, 8:11, 111:211]

print("match =", jnp.allclose(result, expected))

print("result[0, 0, :8] =", result[0, 0, :8])

print("expected[0, 0, :8] =", expected[0, 0, :8])

print(f"\nDumps written to:")

print(f"  HLO:    {HLO_DUMP_PATH}")

print(f"  Mosaic: {MOSAIC_DUMP_PATH}")

print(f"  LLO:    {LLO_DUMP_PATH}")
