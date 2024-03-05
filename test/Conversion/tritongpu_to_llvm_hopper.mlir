// RUN: ENABLE_MMA_V3=1 triton-opt %s -split-input-file --decompose-unsupported-conversions --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=90 2>&1 | FileCheck %s

#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: @dot_high_precision_acc
  tt.func @dot_high_precision_acc(%a: tensor<128x128xf8E5M2, #shared>, %b: tensor<128x256xf8E5M2, #shared1>, %c: tensor<128x256xf32, #mma>) {
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    %m = triton_nvidia_gpu.dot_async %a, %b, %c
      {maxNumImpreciseAcc = 32 : i32, allowTF32 = true} :
      tensor<128x128xf8E5M2, #shared> * tensor<128x256xf8E5M2, #shared1> -> tensor<128x256xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: @dot_low_precision_acc
  tt.func @dot_low_precision_acc(%a: tensor<128x128xf8E5M2, #shared>, %b: tensor<128x256xf8E5M2, #shared1>, %c: tensor<128x256xf32, #mma>) {
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: llvm.return
    %m = triton_nvidia_gpu.dot_async %a, %b, %c
      {maxNumImpreciseAcc = 129 : i32, allowTF32 = true} :
      tensor<128x128xf8E5M2, #shared> * tensor<128x256xf8E5M2, #shared1> -> tensor<128x256xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: @dot_mix_precision_acc
  tt.func @dot_mix_precision_acc(%a: tensor<128x128xf8E5M2, #shared>, %b: tensor<128x256xf8E5M2, #shared1>, %c: tensor<128x256xf32, #mma>) {
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: llvm.return
    %m = triton_nvidia_gpu.dot_async %a, %b, %c
      {maxNumImpreciseAcc = 64 : i32, allowTF32 = true} :
      tensor<128x128xf8E5M2, #shared> * tensor<128x256xf8E5M2, #shared1> -> tensor<128x256xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @dot_zero_acc
  // Generate a wgmma with 2 sources.
  // CHECK: nvgpu.wgmma %{{.*}}, %{{.*}} {
  tt.func @dot_zero_acc(%a: tensor<128x64xf16, #shared>, %b: tensor<64x64xf16, #shared1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %m = triton_nvidia_gpu.dot_async %a, %b, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} :
      tensor<128x64xf16, #shared> * tensor<64x64xf16, #shared1> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @dot_reg_operand_A
  // Generate a wgmma where the first operand is a struct.
  // CHECK: nvgpu.wgmma {{.*}} : (!llvm.struct<(i32, i32, i32, i32)>, i64, !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  // CHECK: nvgpu.wgmma_wait_group %{{.*}} {pendings = 0 : i32} : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  tt.func @dot_reg_operand_A(%a: tensor<128x64xf16, #mma>, %b: tensor<64x64xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %opA = triton_gpu.convert_layout %a : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
    %m = tt.dot %opA, %b, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} :
      tensor<128x64xf16,  #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<64x64xf16, #shared> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 32]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @dot_reg_operand_A_fp8
  // Generate a wgmma where the first operand is a struct.
  // CHECK: nvgpu.wgmma {{.*}} : (!llvm.struct<(i32, i32, i32, i32)>, i64) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  // CHECK: nvgpu.wgmma_wait_group %{{.*}} {pendings = 0 : i32}
  tt.func @dot_reg_operand_A_fp8(%a: tensor<128x128xf8E5M2, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>, %b: tensor<128x256xf8E5M2, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma1>
    %m = tt.dot %a, %b, %cst {allowTF32 = true, maxNumImpreciseAcc = 1073741824 : i32} :
      tensor<128x128xf8E5M2, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<128x256xf8E5M2, #shared> -> tensor<128x256xf32, #mma1>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: test_fp8_to_f16_conversion
  tt.func @test_fp8_to_f16_conversion(
    %in0: tensor<128xf8E5M2, #blocked>, %in1: tensor<128xf8E4M3FNUZ, #blocked>,
    %in2: tensor<128xf16, #blocked>, %in3: tensor<128xf32, #blocked>) {
    // CHECK-COUNT-2: cvt.rn.f16x2.e5m2x2 {{.*}} "=r,h" %{{.*}} : (i16) -> vector<2xf16>
    %out0 = tt.fp_to_fp %in0 : tensor<128xf8E5M2, #blocked> -> tensor<128xf16, #blocked>
    // CHECK-COUNT-2: cvt.rn.f16x2.e4m3x2 {{.*}} "=r,h" %{{.*}} : (i16) -> vector<2xf16>
    %out1 = tt.fp_to_fp %in1 : tensor<128xf8E4M3FNUZ, #blocked> -> tensor<128xf16, #blocked>
    // CHECK-COUNT-2: mul.rn.bf16x2
    %out2 = tt.fp_to_fp %in0 : tensor<128xf8E5M2, #blocked> -> tensor<128xbf16, #blocked>

    // CHECK-COUNT-2: cvt.rn.satfinite.e5m2x2.f16x2 {{.*}} "=h,r" %{{.*}} : (i32) -> vector<2xi8>
    %out3 = tt.fp_to_fp %in2 {rounding = 1 : i32} : tensor<128xf16, #blocked> -> tensor<128xf8E5M2, #blocked>
    // CHECK-COUNT-2: cvt.rn.satfinite.e4m3x2.f16x2 {{.*}} "=h,r" %{{.*}} : (i32) -> vector<2xi8>
    %out4 = tt.fp_to_fp %in2 {rounding = 1 : i32} : tensor<128xf16, #blocked> -> tensor<128xf8E4M3FNUZ, #blocked>

    // CHECK-COUNT-2: cvt.rn.satfinite.e5m2x2.f32 {{.*}} "=h,r,r" %{{.*}}, %{{.*}} : (i32, i32) -> vector<2xi8>
    %out5 = tt.fp_to_fp %in3 {rounding = 1 : i32} : tensor<128xf32, #blocked> -> tensor<128xf8E5M2, #blocked>
    // CHECK-COUNT-2: cvt.rn.satfinite.e4m3x2.f32 {{.*}} "=h,r,r" %{{.*}}, %{{.*}} : (i32, i32) -> vector<2xi8>
    %out6 = tt.fp_to_fp %in3 {rounding = 1 : i32} : tensor<128xf32, #blocked> -> tensor<128xf8E4M3FNUZ, #blocked>
    tt.return
  }
}

// -----

#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], hasLeadingOffset = true}>
// CHECK-LABEL: splat_shared_layout
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @splat_shared_layout(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}) {
    %0 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
    %1 = tt.splat %0 : f32 -> tensor<64x128xf32, #shared>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
// CHECK-LABEL: clamp
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @clamp(%x : tensor<1024xf32, #blocked>, %limit : tensor<1024xf32, #blocked>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
    %neg_limit = arith.subf %cst, %limit : tensor<1024xf32, #blocked>

    // CHECK: "min.xorsign.abs.f32 $0, $1, $2;", "=f,f,f"
    %12 = tt.clampf %x, %neg_limit, %limit {propagateNan = 0 : i32} : tensor<1024xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 16]}>
// CHECK-LABEL: convert_mma_to_blocked
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @convert_mma_to_blocked(%a: tensor<128x256xf16, #mma>) {
    // CHECK-COUNT-16: nvgpu.stmatrix
    //          CHECK: nvvm.barrier0
    %c = triton_gpu.convert_layout %a : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    tt.return
  }
}
