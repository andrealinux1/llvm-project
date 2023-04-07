// RUN: mlir-opt -restructure-clift %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
  llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @foo(%arg0: i32, %arg1: i32) attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(7 : i32) : i32
    %2 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.constant(10 : i32) : i32
    %5 = llvm.mlir.constant(5 : i32) : i32
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %6 : i32, !llvm.ptr
    llvm.store %arg1, %7 : i32, !llvm.ptr
    %8 = llvm.load %6 : !llvm.ptr -> i32
    %9 = llvm.icmp "ne" %8, %1 : i32
    llvm.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb7
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb2, ^bb8
    %10 = llvm.load %6 : !llvm.ptr -> i32
    %11 = llvm.icmp "slt" %10, %4 : i32
    llvm.cond_br %11, ^bb4, ^bb9
  ^bb4:  // pred: ^bb3
    %12 = llvm.load %6 : !llvm.ptr -> i32
    %13 = llvm.icmp "eq" %12, %5 : i32
    llvm.cond_br %13, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %14 = llvm.load %6 : !llvm.ptr -> i32
    %15 = llvm.call @printf(%3, %14) : (!llvm.ptr, i32) -> i32
    llvm.br ^bb8
  ^bb6:  // pred: ^bb4
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb1, ^bb6
    %16 = llvm.load %7 : !llvm.ptr -> i32
    %17 = llvm.call @printf(%3, %16) : (!llvm.ptr, i32) -> i32
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb5, ^bb7
    %18 = llvm.load %6 : !llvm.ptr -> i32
    %19 = llvm.add %18, %0  : i32
    llvm.store %19, %6 : i32, !llvm.ptr
    %20 = llvm.load %7 : !llvm.ptr -> i32
    %21 = llvm.add %20, %0  : i32
    llvm.store %21, %7 : i32, !llvm.ptr
    llvm.br ^bb3
  ^bb9:  // pred: ^bb3
    llvm.return
  }
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

// CHECK: module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
// CHECK-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-NEXT:   llvm.func @foo(%arg0: i32, %arg1: i32) attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
// CHECK-NEXT:     %0 = clift.make_label !clift.label
// CHECK-NEXT:     %1 = clift.make_label !clift.label
// CHECK-NEXT:     %2 = clift.make_label !clift.label
// CHECK-NEXT:     %3 = clift.make_label !clift.label
// CHECK-NEXT:     %4 = clift.make_label !clift.label
// CHECK-NEXT:     %5 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:     %6 = llvm.mlir.constant(7 : i32) : i32
// CHECK-NEXT:     %7 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
// CHECK-NEXT:     %8 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-NEXT:     %9 = llvm.mlir.constant(10 : i32) : i32
// CHECK-NEXT:     %10 = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT:     %11 = llvm.alloca %5 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:     %12 = llvm.alloca %5 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:     llvm.store %arg0, %11 : i32, !llvm.ptr
// CHECK-NEXT:     llvm.store %arg1, %12 : i32, !llvm.ptr
// CHECK-NEXT:     %13 = llvm.load %11 : !llvm.ptr -> i32
// CHECK-NEXT:     %14 = llvm.icmp "ne" %13, %6 : i32
// CHECK-NEXT:     llvm.cond_br %14, ^bb1, ^bb2
// CHECK-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-NEXT:     llvm.br ^bb5
// CHECK-NEXT:   ^bb2:  // pred: ^bb0
// CHECK-NEXT:     llvm.br ^bb6
// CHECK-NEXT:   ^bb3:  // pred: ^bb6
// CHECK-NEXT:     clift.assign_label %0 !clift.label
// CHECK-NEXT:     llvm.br ^bb5
// CHECK-NEXT:   ^bb4:  // 2 preds: ^bb5, ^bb6
// CHECK-NEXT:     clift.assign_label %1 !clift.label
// CHECK-NEXT:     clift.assign_label %2 !clift.label
// CHECK-NEXT:     clift.assign_label %4 !clift.label
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   ^bb5:  // 2 preds: ^bb1, ^bb3
// CHECK-NEXT:     clift.loop {
// CHECK-NEXT:       llvm.br ^bb1
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %15 = llvm.load %12 : !llvm.ptr -> i32
// CHECK-NEXT:       %16 = llvm.call @printf(%8, %15) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:       llvm.br ^bb3
// CHECK-NEXT:     ^bb2:  // pred: ^bb3
// CHECK-NEXT:       clift.loop {
// CHECK-NEXT:         llvm.br ^bb1
// CHECK-NEXT:       ^bb1:  // pred: ^bb0
// CHECK-NEXT:         %21 = llvm.load %11 : !llvm.ptr -> i32
// CHECK-NEXT:         %22 = llvm.icmp "slt" %21, %9 : i32
// CHECK-NEXT:         llvm.cond_br %22, ^bb3, ^bb5
// CHECK-NEXT:       ^bb2:  // pred: ^bb3
// CHECK-NEXT:         %23 = llvm.load %11 : !llvm.ptr -> i32
// CHECK-NEXT:         %24 = llvm.call @printf(%8, %23) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:         llvm.br ^bb4
// CHECK-NEXT:       ^bb3:  // pred: ^bb1
// CHECK-NEXT:         %25 = llvm.load %11 : !llvm.ptr -> i32
// CHECK-NEXT:         %26 = llvm.icmp "eq" %25, %10 : i32
// CHECK-NEXT:         llvm.cond_br %26, ^bb2, ^bb6
// CHECK-NEXT:       ^bb4:  // pred: ^bb2
// CHECK-NEXT:         %27 = llvm.load %11 : !llvm.ptr -> i32
// CHECK-NEXT:         %28 = llvm.add %27, %5  : i32
// CHECK-NEXT:         llvm.store %28, %11 : i32, !llvm.ptr
// CHECK-NEXT:         %29 = llvm.load %12 : !llvm.ptr -> i32
// CHECK-NEXT:         %30 = llvm.add %29, %5  : i32
// CHECK-NEXT:         llvm.store %30, %12 : i32, !llvm.ptr
// CHECK-NEXT:         llvm.br ^bb7
// CHECK-NEXT:       ^bb5:  // pred: ^bb1
// CHECK-NEXT:         clift.goto %4 !clift.label
// CHECK-NEXT:       ^bb6:  // pred: ^bb3
// CHECK-NEXT:         clift.goto %3 !clift.label
// CHECK-NEXT:       ^bb7:  // pred: ^bb4
// CHECK-NEXT:         "clift.continue"() : () -> ()
// CHECK-NEXT:       } ^bb5, ^bb4
// CHECK-NEXT:     ^bb3:  // pred: ^bb1
// CHECK-NEXT:       %17 = llvm.load %11 : !llvm.ptr -> i32
// CHECK-NEXT:       %18 = llvm.add %17, %5  : i32
// CHECK-NEXT:       llvm.store %18, %11 : i32, !llvm.ptr
// CHECK-NEXT:       %19 = llvm.load %12 : !llvm.ptr -> i32
// CHECK-NEXT:       %20 = llvm.add %19, %5  : i32
// CHECK-NEXT:       llvm.store %20, %12 : i32, !llvm.ptr
// CHECK-NEXT:       llvm.br ^bb2
// CHECK-NEXT:     ^bb4:  // pred: ^bb2
// CHECK-NEXT:       clift.assign_label %3 !clift.label
// CHECK-NEXT:       llvm.br ^bb6
// CHECK-NEXT:     ^bb5:  // pred: ^bb2
// CHECK-NEXT:       clift.goto %2 !clift.label
// CHECK-NEXT:     ^bb6:  // pred: ^bb4
// CHECK-NEXT:       "clift.continue"() : () -> ()
// CHECK-NEXT:     } ^bb4
// CHECK-NEXT:   ^bb6:  // pred: ^bb2
// CHECK-NEXT:     clift.loop {
// CHECK-NEXT:       llvm.br ^bb1
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %15 = llvm.load %11 : !llvm.ptr -> i32
// CHECK-NEXT:       %16 = llvm.icmp "slt" %15, %9 : i32
// CHECK-NEXT:       llvm.cond_br %16, ^bb3, ^bb5
// CHECK-NEXT:     ^bb2:  // pred: ^bb3
// CHECK-NEXT:       %17 = llvm.load %11 : !llvm.ptr -> i32
// CHECK-NEXT:       %18 = llvm.call @printf(%8, %17) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:       llvm.br ^bb4
// CHECK-NEXT:     ^bb3:  // pred: ^bb1
// CHECK-NEXT:       %19 = llvm.load %11 : !llvm.ptr -> i32
// CHECK-NEXT:       %20 = llvm.icmp "eq" %19, %10 : i32
// CHECK-NEXT:       llvm.cond_br %20, ^bb2, ^bb6
// CHECK-NEXT:     ^bb4:  // pred: ^bb2
// CHECK-NEXT:       %21 = llvm.load %11 : !llvm.ptr -> i32
// CHECK-NEXT:       %22 = llvm.add %21, %5  : i32
// CHECK-NEXT:       llvm.store %22, %11 : i32, !llvm.ptr
// CHECK-NEXT:       %23 = llvm.load %12 : !llvm.ptr -> i32
// CHECK-NEXT:       %24 = llvm.add %23, %5  : i32
// CHECK-NEXT:       llvm.store %24, %12 : i32, !llvm.ptr
// CHECK-NEXT:       llvm.br ^bb7
// CHECK-NEXT:     ^bb5:  // pred: ^bb1
// CHECK-NEXT:       clift.goto %1 !clift.label
// CHECK-NEXT:     ^bb6:  // pred: ^bb3
// CHECK-NEXT:       clift.goto %0 !clift.label
// CHECK-NEXT:     ^bb7:  // pred: ^bb4
// CHECK-NEXT:       "clift.continue"() : () -> ()
// CHECK-NEXT:     } ^bb4, ^bb3
// CHECK-NEXT:   }
// CHECK-COM:   llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
// CHECK-NEXT: }
