// Mosaic IR kernel — C = A @ B, with A (3x3) and B (3x3) -> C (3x3), f32.
//
// Plain rank-N VMEM memrefs with NO #tpu.tiled layout and NO
// erase_memref_layout, like the modular repo's Mosaic dumps; with
// needs_layout_passes=true in the custom-call config, infer-memref-layout
// assigns the native (8,128) f32 tiling. All dims (M=K=N=3) are sub-tile and
// get padded to the (8,128) tile by Mosaic.
//
// Standard contraction: lhs dim1 (K) with rhs dim0 (K), output MxN.

module {
  func.func @main(
      %lhs: memref<3x3xf32, #tpu.memory_space<vmem>>,   // A (M x K)
      %rhs: memref<3x3xf32, #tpu.memory_space<vmem>>,   // B (K x N)
      %dst: memref<3x3xf32, #tpu.memory_space<vmem>>    // C (M x N)
  ) attributes {tpu.core_type = #tpu.core_type<tc>} {
    %cst = arith.constant dense<0.000000e+00> : vector<3x3xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.load %lhs[%c0,%c0] : memref<3x3xf32, #tpu.memory_space<vmem>>, vector<3x3xf32>
    %1 = vector.load %rhs[%c0,%c0] : memref<3x3xf32, #tpu.memory_space<vmem>>, vector<3x3xf32>
    %2 = tpu.matmul %0, %1, %cst {transpose_lhs_hint = false}
        : vector<3x3xf32>, vector<3x3xf32>, vector<3x3xf32> -> vector<3x3xf32>
    vector.store %2, %dst[%c0,%c0] : memref<3x3xf32, #tpu.memory_space<vmem>>, vector<3x3xf32>
    return
  }
}
