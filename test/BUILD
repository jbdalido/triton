# copybara:uncomment_begin
# load("//third_party/llvm/build_defs:lit.bzl", "glob_lit_tests")
# load("//tools/build_defs/build_test:build_test.bzl", "build_test")
# 
# package(
#     default_applicable_licenses = ["//:license"],
#     default_compatible_with = ["//buildenv/target:non_prod"],
#     default_visibility = ["//:__subpackages__"],
# )
# 
# glob_lit_tests(
#     name = "all_tests",
#     data = [
#         "@llvm-project//llvm:FileCheck",
#         "@llvm-project//llvm:opt",
#         "@llvm-project//mlir:mlir-translate",
#         "//:triton-llvm-opt",
#         "//:triton-opt",
#         "//:triton-tensor-layout",
#     ],
#     driver = "@llvm-project//mlir:run_lit.sh",
#     exclude = [
#         # broken, offending change reverted in
#         # https://github.com/triton-lang/triton/commit/3ed479f2f91a1d94dacb547115d357f5ce3219d8
#         "Conversion/reduce_to_llvm.mlir",
#         "Conversion/amd/dedup-by-constancy.mlir",  # AMD-specific, broken
#         "TritonGPU/amd/amd-instruction-sched.mlir",  # AMD-specific, broken with -debug-only.
#         "TritonGPU/optimize_epilogue.mlir",  # TODO: b/346283526 - AMD-specific, triggering UBSAN
#     ],
#     test_file_exts = [
#         "mlir",
#         "ll",
#     ],
# )
# 
# build_test(
#     name = "build_test",
#     allow_empty_target = False,
#     targets = [
#         "//:TritonDialects",
#         "//:TritonGPUToLLVM",
#         "//:TritonGPUTransforms",
#         "//:TritonLLVMIR",
#         "//:TritonPTX",
#         "//:TritonToTritonGPU",
#         "//:TritonTools",
#         "//:TritonTransforms",
#         "//:triton-opt",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "TritonTestAnalysis",
    srcs = glob(["lib/Analysis/*.cpp"]),
    deps = [
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "//:TritonDialects",
        "@triton//third_party/nvidia:TritonNVIDIAGPUToLLVM",
    ],
)
