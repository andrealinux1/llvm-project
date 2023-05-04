// RUN: mlir-opt -restructure-clift %s | FileCheck %s -check-prefix=CHECK-RESTRUCTURE

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
  llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @foo(%arg0: i32, %arg1: i32) attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(10 : i32) : i32
    %2 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %4 : i32, !llvm.ptr
    llvm.store %arg1, %5 : i32, !llvm.ptr
    %6 = llvm.load %4 : !llvm.ptr -> i32
    %7 = llvm.icmp "slt" %6, %1 : i32
    llvm.cond_br %7, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %8 = llvm.load %5 : !llvm.ptr -> i32
    %9 = llvm.icmp "slt" %8, %1 : i32
    llvm.cond_br %9, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %10 = llvm.load %4 : !llvm.ptr -> i32
    %11 = llvm.call @printf(%3, %10) : (!llvm.ptr, i32) -> i32
    %12 = llvm.load %4 : !llvm.ptr -> i32
    %13 = llvm.add %12, %0  : i32
    llvm.store %13, %4 : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb3:  // 2 preds: ^bb0, ^bb1
    %14 = llvm.load %5 : !llvm.ptr -> i32
    %15 = llvm.call @printf(%3, %14) : (!llvm.ptr, i32) -> i32
    %16 = llvm.load %5 : !llvm.ptr -> i32
    %17 = llvm.add %16, %0  : i32
    llvm.store %17, %5 : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    llvm.return
  }
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

// CHECK-RESTRUCTURE-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
// CHECK-RESTRUCTURE-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-RESTRUCTURE-NEXT:   llvm.func @foo(%arg0: i32, %arg1: i32) attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
// CHECK-RESTRUCTURE-NEXT:     %0 = llvm.mlir.constant(1 : i32) : i32
// CHECK-RESTRUCTURE-NEXT:     %1 = llvm.mlir.constant(10 : i32) : i32
// CHECK-RESTRUCTURE-NEXT:     %2 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
// CHECK-RESTRUCTURE-NEXT:     %3 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.store %arg0, %4 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.store %arg1, %5 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     %6 = llvm.load %4 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %7 = llvm.icmp "slt" %6, %1 : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.cond_br %7, ^bb1, ^bb3
// CHECK-RESTRUCTURE-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-RESTRUCTURE-NEXT:     %8 = llvm.load %5 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %9 = llvm.icmp "slt" %8, %1 : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.cond_br %9, ^bb2, ^bb3
// CHECK-RESTRUCTURE-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-RESTRUCTURE-NEXT:     %10 = llvm.load %4 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %11 = llvm.call @printf(%3, %10) : (!llvm.ptr, i32) -> i32
// CHECK-RESTRUCTURE-NEXT:     %12 = llvm.load %4 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %13 = llvm.add %12, %0  : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.store %13, %4 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb4
// CHECK-RESTRUCTURE-NEXT:   ^bb3:  // 2 preds: ^bb0, ^bb1
// CHECK-RESTRUCTURE-NEXT:     %14 = llvm.load %5 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %15 = llvm.call @printf(%3, %14) : (!llvm.ptr, i32) -> i32
// CHECK-RESTRUCTURE-NEXT:     %16 = llvm.load %5 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %17 = llvm.add %16, %0  : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.store %17, %5 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb4
// CHECK-RESTRUCTURE-NEXT:   ^bb4:  // 2 preds: ^bb2, ^bb3
// CHECK-RESTRUCTURE-NEXT:     llvm.return
// CHECK-RESTRUCTURE-NEXT:   }
// CHECK-RESTRUCTURE-COM:   llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
// CHECK-RESTRUCTURE-NEXT: }
