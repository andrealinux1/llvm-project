// RUN: mlir-opt -restructure-clift %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
  llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @main() attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.constant(10 : i32) : i32
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %5 : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    %6 = llvm.load %5 : !llvm.ptr -> i32
    %7 = llvm.call @printf(%3, %6) : (!llvm.ptr, i32) -> i32
    %8 = llvm.load %5 : !llvm.ptr -> i32
    %9 = llvm.add %8, %0  : i32
    llvm.store %9, %5 : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // pred: ^bb1
    %10 = llvm.load %5 : !llvm.ptr -> i32
    %11 = llvm.icmp "slt" %10, %4 : i32
    llvm.cond_br %11, ^bb1, ^bb3
  ^bb3:  // pred: ^bb2
    llvm.return
  }
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

// CHECK: module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
// CHECK-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-NEXT:   llvm.func @main() attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
// CHECK-NEXT:     %0 = clift.make_label !clift.label
// CHECK-NEXT:     %1 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:     %2 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:     %3 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
// CHECK-NEXT:     %4 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-NEXT:     %5 = llvm.mlir.constant(10 : i32) : i32
// CHECK-NEXT:     %6 = llvm.alloca %1 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:     llvm.store %2, %6 : i32, !llvm.ptr
// CHECK-NEXT:     llvm.br ^bb2
// CHECK-NEXT:   ^bb1:  // pred: ^bb2
// CHECK-NEXT:     clift.assign_label %0 !clift.label
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   ^bb2:  // pred: ^bb0
// CHECK-NEXT:     clift.loop {
// CHECK-NEXT:       llvm.br ^bb1
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %7 = llvm.load %6 : !llvm.ptr -> i32
// CHECK-NEXT:       %8 = llvm.call @printf(%4, %7) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:       %9 = llvm.load %6 : !llvm.ptr -> i32
// CHECK-NEXT:       %10 = llvm.add %9, %1  : i32
// CHECK-NEXT:       llvm.store %10, %6 : i32, !llvm.ptr
// CHECK-NEXT:       llvm.br ^bb2
// CHECK-NEXT:     ^bb2:  // pred: ^bb1
// CHECK-NEXT:       %11 = llvm.load %6 : !llvm.ptr -> i32
// CHECK-NEXT:       %12 = llvm.icmp "slt" %11, %5 : i32
// CHECK-NEXT:       llvm.cond_br %12, ^bb4, ^bb3
// CHECK-NEXT:     ^bb3:  // pred: ^bb2
// CHECK-NEXT:       clift.goto %0 !clift.label
// CHECK-NEXT:     ^bb4:  // pred: ^bb2
// CHECK-NEXT:       "clift.continue"() : () -> ()
// CHECK-NEXT:     } ^bb1
// CHECK-NEXT:   }
// CHECK-COM:   llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
// CHECK-NEXT: }
