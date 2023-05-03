// RUN: mlir-opt -restructure-clift %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
  llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @foo(%arg0: i32, %arg1: i32) attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.addressof @".str" : !llvm.ptr
    %3 = llvm.mlir.constant(5 : i32) : i32
    %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %4 : i32, !llvm.ptr
    llvm.store %arg1, %5 : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb4
    %6 = llvm.load %4 : !llvm.ptr -> i32
    %7 = llvm.call @printf(%2, %6) : (!llvm.ptr, i32) -> i32
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb7
    %8 = llvm.load %5 : !llvm.ptr -> i32
    %9 = llvm.call @printf(%2, %8) : (!llvm.ptr, i32) -> i32
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %10 = llvm.load %4 : !llvm.ptr -> i32
    %11 = llvm.icmp "eq" %10, %3 : i32
    llvm.cond_br %11, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.br ^bb1
  ^bb5:  // pred: ^bb3
    llvm.br ^bb6
  ^bb6:  // pred: ^bb5
    %12 = llvm.load %5 : !llvm.ptr -> i32
    %13 = llvm.icmp "eq" %12, %3 : i32
    llvm.cond_br %13, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.br ^bb2
  ^bb8:  // pred: ^bb6
    llvm.return
  }
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

// CHECK-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
// CHECK-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-NEXT:   llvm.func @foo(%arg0: i32, %arg1: i32) attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
// CHECK-NEXT:     %0 = clift.make_label !clift.label
// CHECK-NEXT:     %1 = clift.make_label !clift.label
// CHECK-NEXT:     %2 = clift.make_label !clift.label
// CHECK-NEXT:     %3 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:     %4 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
// CHECK-NEXT:     %5 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-NEXT:     %6 = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT:     %7 = llvm.alloca %3 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:     %8 = llvm.alloca %3 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:     llvm.store %arg0, %7 : i32, !llvm.ptr
// CHECK-NEXT:     llvm.store %arg1, %8 : i32, !llvm.ptr
// CHECK-NEXT:     llvm.br ^bb2
// CHECK-NEXT:   ^bb1:  // pred: ^bb2
// CHECK-NEXT:     clift.assign_label %0 !clift.label
// CHECK-NEXT:     clift.assign_label %1 !clift.label
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   ^bb2:  // pred: ^bb0
// CHECK-NEXT:     clift.loop {
// CHECK-NEXT:       llvm.br ^bb1
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %9 = llvm.load %7 : !llvm.ptr -> i32
// CHECK-NEXT:       %10 = llvm.call @printf(%5, %9) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:       llvm.br ^bb2
// CHECK-NEXT:     ^bb2:  // pred: ^bb1
// CHECK-NEXT:       clift.loop {
// CHECK-NEXT:         llvm.br ^bb1
// CHECK-NEXT:       ^bb1:  // pred: ^bb0
// CHECK-NEXT:         %11 = llvm.load %8 : !llvm.ptr -> i32
// CHECK-NEXT:         %12 = llvm.call @printf(%5, %11) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:         llvm.br ^bb4
// CHECK-NEXT:       ^bb2:  // pred: ^bb3
// CHECK-NEXT:         %13 = llvm.load %8 : !llvm.ptr -> i32
// CHECK-NEXT:         %14 = llvm.icmp "eq" %13, %6 : i32
// CHECK-NEXT:         llvm.cond_br %14, ^bb5, ^bb7
// CHECK-NEXT:       ^bb3:  // pred: ^bb4
// CHECK-NEXT:         llvm.br ^bb2
// CHECK-NEXT:       ^bb4:  // pred: ^bb1
// CHECK-NEXT:         %15 = llvm.load %7 : !llvm.ptr -> i32
// CHECK-NEXT:         %16 = llvm.icmp "eq" %15, %6 : i32
// CHECK-NEXT:         llvm.cond_br %16, ^bb6, ^bb3
// CHECK-NEXT:       ^bb5:  // pred: ^bb2
// CHECK-NEXT:         llvm.br ^bb8
// CHECK-NEXT:       ^bb6:  // pred: ^bb4
// CHECK-NEXT:         clift.goto %2 !clift.label
// CHECK-NEXT:       ^bb7:  // pred: ^bb2
// CHECK-NEXT:         clift.goto %1 !clift.label
// CHECK-NEXT:       ^bb8:  // pred: ^bb5
// CHECK-NEXT:         "clift.continue"() : () -> ()
// CHECK-NEXT:       } ^bb3, ^bb4
// CHECK-NEXT:     ^bb3:  // pred: ^bb2
// CHECK-NEXT:       clift.assign_label %2 !clift.label
// CHECK-NEXT:       llvm.br ^bb5
// CHECK-NEXT:     ^bb4:  // pred: ^bb2
// CHECK-NEXT:       clift.goto %0 !clift.label
// CHECK-NEXT:     ^bb5:  // pred: ^bb3
// CHECK-NEXT:       "clift.continue"() : () -> ()
// CHECK-NEXT:     } ^bb1
// CHECK-NEXT:   }
// CHECK-COM:   llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
// CHECK-NEXT: }
