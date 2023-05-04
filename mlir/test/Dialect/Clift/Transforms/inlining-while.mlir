// RUN: mlir-opt -restructure-clift %s | FileCheck %s -check-prefix=CHECK-RESTRUCTURE
// RUN: mlir-opt -restructure-clift -comb-clift %s | FileCheck %s -check-prefix=CHECK-COMB

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
  llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @main() attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(10 : i32) : i32
    %3 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.addressof @".str" : !llvm.ptr
    %5 = llvm.mlir.constant(5 : i32) : i32
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %6 : i32, !llvm.ptr
    llvm.store %1, %7 : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb8
    %8 = llvm.load %6 : !llvm.ptr -> i32
    %9 = llvm.icmp "slt" %8, %2 : i32
    llvm.cond_br %9, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %10 = llvm.load %6 : !llvm.ptr -> i32
    %11 = llvm.call @printf(%4, %10) : (!llvm.ptr, i32) -> i32
    %12 = llvm.load %6 : !llvm.ptr -> i32
    %13 = llvm.add %12, %0  : i32
    llvm.store %13, %6 : i32, !llvm.ptr
    %14 = llvm.load %6 : !llvm.ptr -> i32
    %15 = llvm.icmp "eq" %14, %5 : i32
    llvm.cond_br %15, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.br ^bb7
  ^bb4:  // pred: ^bb2
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb4, ^bb7
    %16 = llvm.load %7 : !llvm.ptr -> i32
    %17 = llvm.icmp "slt" %16, %2 : i32
    llvm.cond_br %17, ^bb6, ^bb8
  ^bb6:  // pred: ^bb5
    %18 = llvm.load %7 : !llvm.ptr -> i32
    %19 = llvm.call @printf(%4, %18) : (!llvm.ptr, i32) -> i32
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb3, ^bb6
    %20 = llvm.load %7 : !llvm.ptr -> i32
    %21 = llvm.add %20, %0  : i32
    llvm.store %21, %7 : i32, !llvm.ptr
    llvm.br ^bb5
  ^bb8:  // pred: ^bb5
    llvm.br ^bb1
  ^bb9:  // pred: ^bb1
    llvm.return
  }
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

// CHECK-RESTRUCTURE-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
// CHECK-RESTRUCTURE-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-RESTRUCTURE-NEXT:   llvm.func @main() attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
// CHECK-RESTRUCTURE-NEXT:     %0 = clift.make_label !clift.label
// CHECK-RESTRUCTURE-NEXT:     %1 = clift.make_label !clift.label
// CHECK-RESTRUCTURE-NEXT:     %2 = clift.make_label !clift.label
// CHECK-RESTRUCTURE-NEXT:     %3 = llvm.mlir.constant(1 : i32) : i32
// CHECK-RESTRUCTURE-NEXT:     %4 = llvm.mlir.constant(0 : i32) : i32
// CHECK-RESTRUCTURE-NEXT:     %5 = llvm.mlir.constant(10 : i32) : i32
// CHECK-RESTRUCTURE-NEXT:     %6 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
// CHECK-RESTRUCTURE-NEXT:     %7 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     %8 = llvm.mlir.constant(5 : i32) : i32
// CHECK-RESTRUCTURE-NEXT:     %9 = llvm.alloca %3 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     %10 = llvm.alloca %3 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.store %4, %9 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.store %4, %10 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb14
// CHECK-RESTRUCTURE-NEXT:   ^bb1:  // 2 preds: ^bb11, ^bb12
// CHECK-RESTRUCTURE-NEXT:     %11 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %12 = llvm.icmp "slt" %11, %5 : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.cond_br %12, ^bb2, ^bb13
// CHECK-RESTRUCTURE-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-RESTRUCTURE-NEXT:     %13 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %14 = llvm.call @printf(%7, %13) : (!llvm.ptr, i32) -> i32
// CHECK-RESTRUCTURE-NEXT:     %15 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %16 = llvm.add %15, %3  : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.store %16, %9 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     %17 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %18 = llvm.icmp "eq" %17, %8 : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.cond_br %18, ^bb3, ^bb4
// CHECK-RESTRUCTURE-NEXT:   ^bb3:  // pred: ^bb2
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb9
// CHECK-RESTRUCTURE-NEXT:   ^bb4:  // pred: ^bb2
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb6
// CHECK-RESTRUCTURE-NEXT:   ^bb5:  // pred: ^bb9
// CHECK-RESTRUCTURE-NEXT:     %19 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %20 = llvm.icmp "slt" %19, %5 : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.cond_br %20, ^bb7, ^bb11
// CHECK-RESTRUCTURE-NEXT:   ^bb6:  // 2 preds: ^bb4, ^bb10
// CHECK-RESTRUCTURE-NEXT:     %21 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %22 = llvm.icmp "slt" %21, %5 : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.cond_br %22, ^bb8, ^bb12
// CHECK-RESTRUCTURE-NEXT:   ^bb7:  // pred: ^bb5
// CHECK-RESTRUCTURE-NEXT:     %23 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %24 = llvm.call @printf(%7, %23) : (!llvm.ptr, i32) -> i32
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb9
// CHECK-RESTRUCTURE-NEXT:   ^bb8:  // pred: ^bb6
// CHECK-RESTRUCTURE-NEXT:     %25 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %26 = llvm.call @printf(%7, %25) : (!llvm.ptr, i32) -> i32
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb10
// CHECK-RESTRUCTURE-NEXT:   ^bb9:  // 2 preds: ^bb3, ^bb7
// CHECK-RESTRUCTURE-NEXT:     %27 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %28 = llvm.add %27, %3  : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.store %28, %10 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb5
// CHECK-RESTRUCTURE-NEXT:   ^bb10:  // pred: ^bb8
// CHECK-RESTRUCTURE-NEXT:     %29 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %30 = llvm.add %29, %3  : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.store %30, %10 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb6
// CHECK-RESTRUCTURE-NEXT:   ^bb11:  // pred: ^bb5
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb1
// CHECK-RESTRUCTURE-NEXT:   ^bb12:  // pred: ^bb6
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb1
// CHECK-RESTRUCTURE-NEXT:   ^bb13:  // 2 preds: ^bb1, ^bb14
// CHECK-RESTRUCTURE-NEXT:     clift.assign_label %0 !clift.label
// CHECK-RESTRUCTURE-NEXT:     llvm.return
// CHECK-RESTRUCTURE-NEXT:   ^bb14:  // pred: ^bb0
// CHECK-RESTRUCTURE-NEXT:     clift.loop {
// CHECK-RESTRUCTURE-NEXT:       llvm.br ^bb1
// CHECK-RESTRUCTURE-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-RESTRUCTURE-NEXT:       %31 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:       %32 = llvm.icmp "slt" %31, %5 : i32
// CHECK-RESTRUCTURE-NEXT:       llvm.cond_br %32, ^bb7, ^bb9
// CHECK-RESTRUCTURE-NEXT:     ^bb2:  // pred: ^bb6
// CHECK-RESTRUCTURE-NEXT:       clift.loop {
// CHECK-RESTRUCTURE-NEXT:         llvm.br ^bb1
// CHECK-RESTRUCTURE-NEXT:       ^bb1:  // pred: ^bb0
// CHECK-RESTRUCTURE-NEXT:         %39 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:         %40 = llvm.add %39, %3  : i32
// CHECK-RESTRUCTURE-NEXT:         llvm.store %40, %10 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:         llvm.br ^bb2
// CHECK-RESTRUCTURE-NEXT:       ^bb2:  // pred: ^bb1
// CHECK-RESTRUCTURE-NEXT:         %41 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:         %42 = llvm.icmp "slt" %41, %5 : i32
// CHECK-RESTRUCTURE-NEXT:         llvm.cond_br %42, ^bb3, ^bb4
// CHECK-RESTRUCTURE-NEXT:       ^bb3:  // pred: ^bb2
// CHECK-RESTRUCTURE-NEXT:         %43 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:         %44 = llvm.call @printf(%7, %43) : (!llvm.ptr, i32) -> i32
// CHECK-RESTRUCTURE-NEXT:         llvm.br ^bb5
// CHECK-RESTRUCTURE-NEXT:       ^bb4:  // pred: ^bb2
// CHECK-RESTRUCTURE-NEXT:         clift.goto %1 !clift.label
// CHECK-RESTRUCTURE-NEXT:       ^bb5:  // pred: ^bb3
// CHECK-RESTRUCTURE-NEXT:         "clift.continue"() : () -> ()
// CHECK-RESTRUCTURE-NEXT:       } ^bb8
// CHECK-RESTRUCTURE-NEXT:     ^bb3:  // pred: ^bb4
// CHECK-RESTRUCTURE-NEXT:       clift.loop {
// CHECK-RESTRUCTURE-NEXT:         llvm.br ^bb1
// CHECK-RESTRUCTURE-NEXT:       ^bb1:  // pred: ^bb0
// CHECK-RESTRUCTURE-NEXT:         %39 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:         %40 = llvm.icmp "slt" %39, %5 : i32
// CHECK-RESTRUCTURE-NEXT:         llvm.cond_br %40, ^bb2, ^bb4
// CHECK-RESTRUCTURE-NEXT:       ^bb2:  // pred: ^bb1
// CHECK-RESTRUCTURE-NEXT:         %41 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:         %42 = llvm.call @printf(%7, %41) : (!llvm.ptr, i32) -> i32
// CHECK-RESTRUCTURE-NEXT:         llvm.br ^bb3
// CHECK-RESTRUCTURE-NEXT:       ^bb3:  // pred: ^bb2
// CHECK-RESTRUCTURE-NEXT:         %43 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:         %44 = llvm.add %43, %3  : i32
// CHECK-RESTRUCTURE-NEXT:         llvm.store %44, %10 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:         llvm.br ^bb5
// CHECK-RESTRUCTURE-NEXT:       ^bb4:  // pred: ^bb1
// CHECK-RESTRUCTURE-NEXT:         clift.goto %2 !clift.label
// CHECK-RESTRUCTURE-NEXT:       ^bb5:  // pred: ^bb3
// CHECK-RESTRUCTURE-NEXT:         "clift.continue"() : () -> ()
// CHECK-RESTRUCTURE-NEXT:       } ^bb5
// CHECK-RESTRUCTURE-NEXT:     ^bb4:  // pred: ^bb7
// CHECK-RESTRUCTURE-NEXT:       llvm.br ^bb3
// CHECK-RESTRUCTURE-NEXT:     ^bb5:  // pred: ^bb3
// CHECK-RESTRUCTURE-NEXT:       clift.assign_label %2 !clift.label
// CHECK-RESTRUCTURE-NEXT:       llvm.br ^bb10
// CHECK-RESTRUCTURE-NEXT:     ^bb6:  // pred: ^bb7
// CHECK-RESTRUCTURE-NEXT:       llvm.br ^bb2
// CHECK-RESTRUCTURE-NEXT:     ^bb7:  // pred: ^bb1
// CHECK-RESTRUCTURE-NEXT:       %33 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:       %34 = llvm.call @printf(%7, %33) : (!llvm.ptr, i32) -> i32
// CHECK-RESTRUCTURE-NEXT:       %35 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:       %36 = llvm.add %35, %3  : i32
// CHECK-RESTRUCTURE-NEXT:       llvm.store %36, %9 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:       %37 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:       %38 = llvm.icmp "eq" %37, %8 : i32
// CHECK-RESTRUCTURE-NEXT:       llvm.cond_br %38, ^bb6, ^bb4
// CHECK-RESTRUCTURE-NEXT:     ^bb8:  // pred: ^bb2
// CHECK-RESTRUCTURE-NEXT:       clift.assign_label %1 !clift.label
// CHECK-RESTRUCTURE-NEXT:       llvm.br ^bb11
// CHECK-RESTRUCTURE-NEXT:     ^bb9:  // pred: ^bb1
// CHECK-RESTRUCTURE-NEXT:       clift.goto %0 !clift.label
// CHECK-RESTRUCTURE-NEXT:     ^bb10:  // pred: ^bb5
// CHECK-RESTRUCTURE-NEXT:       "clift.continue"() : () -> ()
// CHECK-RESTRUCTURE-NEXT:     ^bb11:  // pred: ^bb8
// CHECK-RESTRUCTURE-NEXT:       "clift.continue"() : () -> ()
// CHECK-RESTRUCTURE-NEXT:     } ^bb13
// CHECK-RESTRUCTURE-NEXT:   }
// CHECK-RESTRUCTURE-COM:   llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
// CHECK-RESTRUCTURE-NEXT: }

// CHECK-COMB-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
// CHECK-COMB-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-COMB-NEXT:   llvm.func @main() attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
// CHECK-COMB-NEXT:     %0 = clift.make_label !clift.label
// CHECK-COMB-NEXT:     %1 = clift.make_label !clift.label
// CHECK-COMB-NEXT:     %2 = clift.make_label !clift.label
// CHECK-COMB-NEXT:     %3 = llvm.mlir.constant(1 : i32) : i32
// CHECK-COMB-NEXT:     %4 = llvm.mlir.constant(0 : i32) : i32
// CHECK-COMB-NEXT:     %5 = llvm.mlir.constant(10 : i32) : i32
// CHECK-COMB-NEXT:     %6 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
// CHECK-COMB-NEXT:     %7 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-COMB-NEXT:     %8 = llvm.mlir.constant(5 : i32) : i32
// CHECK-COMB-NEXT:     %9 = llvm.alloca %3 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-COMB-NEXT:     %10 = llvm.alloca %3 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-COMB-NEXT:     llvm.store %4, %9 : i32, !llvm.ptr
// CHECK-COMB-NEXT:     llvm.store %4, %10 : i32, !llvm.ptr
// CHECK-COMB-NEXT:     llvm.br ^bb14
// CHECK-COMB-NEXT:   ^bb1:  // 2 preds: ^bb11, ^bb12
// CHECK-COMB-NEXT:     %11 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %12 = llvm.icmp "slt" %11, %5 : i32
// CHECK-COMB-NEXT:     llvm.cond_br %12, ^bb2, ^bb13
// CHECK-COMB-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-COMB-NEXT:     %13 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %14 = llvm.call @printf(%7, %13) : (!llvm.ptr, i32) -> i32
// CHECK-COMB-NEXT:     %15 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %16 = llvm.add %15, %3  : i32
// CHECK-COMB-NEXT:     llvm.store %16, %9 : i32, !llvm.ptr
// CHECK-COMB-NEXT:     %17 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %18 = llvm.icmp "eq" %17, %8 : i32
// CHECK-COMB-NEXT:     llvm.cond_br %18, ^bb3, ^bb4
// CHECK-COMB-NEXT:   ^bb3:  // pred: ^bb2
// CHECK-COMB-NEXT:     llvm.br ^bb9
// CHECK-COMB-NEXT:   ^bb4:  // pred: ^bb2
// CHECK-COMB-NEXT:     llvm.br ^bb6
// CHECK-COMB-NEXT:   ^bb5:  // pred: ^bb9
// CHECK-COMB-NEXT:     %19 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %20 = llvm.icmp "slt" %19, %5 : i32
// CHECK-COMB-NEXT:     llvm.cond_br %20, ^bb7, ^bb11
// CHECK-COMB-NEXT:   ^bb6:  // 2 preds: ^bb4, ^bb10
// CHECK-COMB-NEXT:     %21 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %22 = llvm.icmp "slt" %21, %5 : i32
// CHECK-COMB-NEXT:     llvm.cond_br %22, ^bb8, ^bb12
// CHECK-COMB-NEXT:   ^bb7:  // pred: ^bb5
// CHECK-COMB-NEXT:     %23 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %24 = llvm.call @printf(%7, %23) : (!llvm.ptr, i32) -> i32
// CHECK-COMB-NEXT:     llvm.br ^bb9
// CHECK-COMB-NEXT:   ^bb8:  // pred: ^bb6
// CHECK-COMB-NEXT:     %25 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %26 = llvm.call @printf(%7, %25) : (!llvm.ptr, i32) -> i32
// CHECK-COMB-NEXT:     llvm.br ^bb10
// CHECK-COMB-NEXT:   ^bb9:  // 2 preds: ^bb3, ^bb7
// CHECK-COMB-NEXT:     %27 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %28 = llvm.add %27, %3  : i32
// CHECK-COMB-NEXT:     llvm.store %28, %10 : i32, !llvm.ptr
// CHECK-COMB-NEXT:     llvm.br ^bb5
// CHECK-COMB-NEXT:   ^bb10:  // pred: ^bb8
// CHECK-COMB-NEXT:     %29 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %30 = llvm.add %29, %3  : i32
// CHECK-COMB-NEXT:     llvm.store %30, %10 : i32, !llvm.ptr
// CHECK-COMB-NEXT:     llvm.br ^bb6
// CHECK-COMB-NEXT:   ^bb11:  // pred: ^bb5
// CHECK-COMB-NEXT:     llvm.br ^bb1
// CHECK-COMB-NEXT:   ^bb12:  // pred: ^bb6
// CHECK-COMB-NEXT:     llvm.br ^bb1
// CHECK-COMB-NEXT:   ^bb13:  // 2 preds: ^bb1, ^bb14
// CHECK-COMB-NEXT:     clift.assign_label %0 !clift.label
// CHECK-COMB-NEXT:     llvm.return
// CHECK-COMB-NEXT:   ^bb14:  // pred: ^bb0
// CHECK-COMB-NEXT:     clift.loop {
// CHECK-COMB-NEXT:       llvm.br ^bb1
// CHECK-COMB-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-COMB-NEXT:       %31 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:       %32 = llvm.icmp "slt" %31, %5 : i32
// CHECK-COMB-NEXT:       llvm.cond_br %32, ^bb7, ^bb9
// CHECK-COMB-NEXT:     ^bb2:  // pred: ^bb6
// CHECK-COMB-NEXT:       clift.loop {
// CHECK-COMB-NEXT:         llvm.br ^bb1
// CHECK-COMB-NEXT:       ^bb1:  // pred: ^bb0
// CHECK-COMB-NEXT:         %39 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:         %40 = llvm.add %39, %3  : i32
// CHECK-COMB-NEXT:         llvm.store %40, %10 : i32, !llvm.ptr
// CHECK-COMB-NEXT:         llvm.br ^bb2
// CHECK-COMB-NEXT:       ^bb2:  // pred: ^bb1
// CHECK-COMB-NEXT:         %41 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:         %42 = llvm.icmp "slt" %41, %5 : i32
// CHECK-COMB-NEXT:         llvm.cond_br %42, ^bb3, ^bb4
// CHECK-COMB-NEXT:       ^bb3:  // pred: ^bb2
// CHECK-COMB-NEXT:         %43 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:         %44 = llvm.call @printf(%7, %43) : (!llvm.ptr, i32) -> i32
// CHECK-COMB-NEXT:         llvm.br ^bb5
// CHECK-COMB-NEXT:       ^bb4:  // pred: ^bb2
// CHECK-COMB-NEXT:         clift.goto %1 !clift.label
// CHECK-COMB-NEXT:       ^bb5:  // pred: ^bb3
// CHECK-COMB-NEXT:         "clift.continue"() : () -> ()
// CHECK-COMB-NEXT:       } ^bb8
// CHECK-COMB-NEXT:     ^bb3:  // pred: ^bb4
// CHECK-COMB-NEXT:       clift.loop {
// CHECK-COMB-NEXT:         llvm.br ^bb1
// CHECK-COMB-NEXT:       ^bb1:  // pred: ^bb0
// CHECK-COMB-NEXT:         %39 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:         %40 = llvm.icmp "slt" %39, %5 : i32
// CHECK-COMB-NEXT:         llvm.cond_br %40, ^bb2, ^bb4
// CHECK-COMB-NEXT:       ^bb2:  // pred: ^bb1
// CHECK-COMB-NEXT:         %41 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:         %42 = llvm.call @printf(%7, %41) : (!llvm.ptr, i32) -> i32
// CHECK-COMB-NEXT:         llvm.br ^bb3
// CHECK-COMB-NEXT:       ^bb3:  // pred: ^bb2
// CHECK-COMB-NEXT:         %43 = llvm.load %10 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:         %44 = llvm.add %43, %3  : i32
// CHECK-COMB-NEXT:         llvm.store %44, %10 : i32, !llvm.ptr
// CHECK-COMB-NEXT:         llvm.br ^bb5
// CHECK-COMB-NEXT:       ^bb4:  // pred: ^bb1
// CHECK-COMB-NEXT:         clift.goto %2 !clift.label
// CHECK-COMB-NEXT:       ^bb5:  // pred: ^bb3
// CHECK-COMB-NEXT:         "clift.continue"() : () -> ()
// CHECK-COMB-NEXT:       } ^bb5
// CHECK-COMB-NEXT:     ^bb4:  // pred: ^bb7
// CHECK-COMB-NEXT:       llvm.br ^bb3
// CHECK-COMB-NEXT:     ^bb5:  // pred: ^bb3
// CHECK-COMB-NEXT:       clift.assign_label %2 !clift.label
// CHECK-COMB-NEXT:       llvm.br ^bb10
// CHECK-COMB-NEXT:     ^bb6:  // pred: ^bb7
// CHECK-COMB-NEXT:       llvm.br ^bb2
// CHECK-COMB-NEXT:     ^bb7:  // pred: ^bb1
// CHECK-COMB-NEXT:       %33 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:       %34 = llvm.call @printf(%7, %33) : (!llvm.ptr, i32) -> i32
// CHECK-COMB-NEXT:       %35 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:       %36 = llvm.add %35, %3  : i32
// CHECK-COMB-NEXT:       llvm.store %36, %9 : i32, !llvm.ptr
// CHECK-COMB-NEXT:       %37 = llvm.load %9 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:       %38 = llvm.icmp "eq" %37, %8 : i32
// CHECK-COMB-NEXT:       llvm.cond_br %38, ^bb6, ^bb4
// CHECK-COMB-NEXT:     ^bb8:  // pred: ^bb2
// CHECK-COMB-NEXT:       clift.assign_label %1 !clift.label
// CHECK-COMB-NEXT:       llvm.br ^bb11
// CHECK-COMB-NEXT:     ^bb9:  // pred: ^bb1
// CHECK-COMB-NEXT:       clift.goto %0 !clift.label
// CHECK-COMB-NEXT:     ^bb10:  // pred: ^bb5
// CHECK-COMB-NEXT:       "clift.continue"() : () -> ()
// CHECK-COMB-NEXT:     ^bb11:  // pred: ^bb8
// CHECK-COMB-NEXT:       "clift.continue"() : () -> ()
// CHECK-COMB-NEXT:     } ^bb13
// CHECK-COMB-NEXT:   }
// CHECK-COMB-COM:   llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
// CHECK-COMB-NEXT: }
