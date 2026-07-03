// Sample Mosaic IR kernel — vector add on 8x128 f32 VMEM memrefs.
//
// Shape choice: 8x128 matches the TPU VPU's physical register file
// (8 sublanes × 128 lanes). Mosaic's apply_vector_layout pass expects
// vectors that align with this geometry; rank-1 128-element vectors
// fail to legalize because they don't have a 2D shape to map onto the
// sublane dimension.
//
// Layout (libtpu 0.23 syntax):
//     #tpu.tiled<(8,128),[1,1]>          rank=2, single tile (8,128)
//     #tpu.memory_space<vmem>
//
// Inner uses get a layout-erased memref via tpu.erase_memref_layout
// so upstream MLIR's unit-stride check on vector.load is satisfied.

module {
  func.func @main(
      %lhs_tiled: memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>>,
      %rhs_tiled: memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>>,
      %dst_tiled: memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>>
  ) {
    %lhs = tpu.erase_memref_layout %lhs_tiled
        : memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>>
       -> memref<8x128xf32, #tpu.memory_space<vmem>>
    %rhs = tpu.erase_memref_layout %rhs_tiled
        : memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>>
       -> memref<8x128xf32, #tpu.memory_space<vmem>>
    %dst = tpu.erase_memref_layout %dst_tiled
        : memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>>
       -> memref<8x128xf32, #tpu.memory_space<vmem>>

    %c0 = arith.constant 0 : index
    %v_lhs = vector.load %lhs[%c0, %c0]
        : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    %v_rhs = vector.load %rhs[%c0, %c0]
        : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    %v_sum = arith.addf %v_lhs, %v_rhs : vector<8x128xf32>
    vector.store %v_sum, %dst[%c0, %c0]
        : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    return
  }
}
