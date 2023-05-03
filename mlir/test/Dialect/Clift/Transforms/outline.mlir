// RUN: mlir-opt -restructure-clift %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
  llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @foo(%arg0: i32, %arg1: i32, %arg2: i32) attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.addressof @".str" : !llvm.ptr
    %3 = llvm.mlir.constant(10 : i32) : i32
    %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %4 : i32, !llvm.ptr
    llvm.store %arg1, %5 : i32, !llvm.ptr
    llvm.store %arg2, %6 : i32, !llvm.ptr
    %7 = llvm.load %6 : !llvm.ptr -> i32
    %8 = llvm.call @printf(%2, %7) : (!llvm.ptr, i32) -> i32
    %9 = llvm.load %6 : !llvm.ptr -> i32
    %10 = llvm.icmp "eq" %9, %3 : i32
    llvm.cond_br %10, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb4
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb2, ^bb5
    %11 = llvm.load %4 : !llvm.ptr -> i32
    %12 = llvm.call @printf(%2, %11) : (!llvm.ptr, i32) -> i32
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb1, ^bb3
    %13 = llvm.load %5 : !llvm.ptr -> i32
    %14 = llvm.call @printf(%2, %13) : (!llvm.ptr, i32) -> i32
    llvm.br ^bb5
  ^bb5:  // pred: ^bb4
    llvm.br ^bb3
  }
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

// CHECK-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
// CHECK-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-NEXT:   llvm.func @foo(%arg0: i32, %arg1: i32, %arg2: i32) attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
// CHECK-NEXT:     %0 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:     %1 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
// CHECK-NEXT:     %2 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-NEXT:     %3 = llvm.mlir.constant(10 : i32) : i32
// CHECK-NEXT:     %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:     %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:     %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:     llvm.store %arg0, %4 : i32, !llvm.ptr
// CHECK-NEXT:     llvm.store %arg1, %5 : i32, !llvm.ptr
// CHECK-NEXT:     llvm.store %arg2, %6 : i32, !llvm.ptr
// CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr -> i32
// CHECK-NEXT:     %8 = llvm.call @printf(%2, %7) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:     %9 = llvm.load %6 : !llvm.ptr -> i32
// CHECK-NEXT:     %10 = llvm.icmp "eq" %9, %3 : i32
// CHECK-NEXT:     llvm.cond_br %10, ^bb1, ^bb2
// CHECK-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-NEXT:     llvm.br ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb0
// CHECK-NEXT:     llvm.br ^bb5
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     %11 = llvm.load %5 : !llvm.ptr -> i32
// CHECK-NEXT:     %12 = llvm.call @printf(%2, %11) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:     llvm.br ^bb4
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     llvm.br ^bb5
// CHECK-NEXT:   ^bb5:  // 2 preds: ^bb2, ^bb4
// CHECK-NEXT:     clift.loop {
// CHECK-NEXT:       llvm.br ^bb1
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %13 = llvm.load %4 : !llvm.ptr -> i32
// CHECK-NEXT:       %14 = llvm.call @printf(%2, %13) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:       llvm.br ^bb3
// CHECK-NEXT:     ^bb2:  // pred: ^bb3
// CHECK-NEXT:       llvm.br ^bb4
// CHECK-NEXT:     ^bb3:  // pred: ^bb1
// CHECK-NEXT:       %15 = llvm.load %5 : !llvm.ptr -> i32
// CHECK-NEXT:       %16 = llvm.call @printf(%2, %15) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:       llvm.br ^bb2
// CHECK-NEXT:     ^bb4:  // pred: ^bb2
// CHECK-NEXT:       "clift.continue"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-COM:   llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
// CHECK-NEXT: }
