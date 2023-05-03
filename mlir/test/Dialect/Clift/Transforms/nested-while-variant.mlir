// RUN: mlir-opt -restructure-clift %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
  llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @main() attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(10 : i32) : i32
    %3 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.addressof @".str" : !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %5 : i32, !llvm.ptr
    llvm.store %1, %6 : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb5
    %7 = llvm.load %5 : !llvm.ptr -> i32
    %8 = llvm.icmp "slt" %7, %2 : i32
    llvm.cond_br %8, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb2, ^bb4
    %9 = llvm.load %6 : !llvm.ptr -> i32
    %10 = llvm.icmp "slt" %9, %2 : i32
    llvm.cond_br %10, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %11 = llvm.load %6 : !llvm.ptr -> i32
    %12 = llvm.call @printf(%4, %11) : (!llvm.ptr, i32) -> i32
    %13 = llvm.load %6 : !llvm.ptr -> i32
    %14 = llvm.add %13, %0  : i32
    llvm.store %14, %6 : i32, !llvm.ptr
    llvm.br ^bb3
  ^bb5:  // pred: ^bb3
    %15 = llvm.load %5 : !llvm.ptr -> i32
    %16 = llvm.call @printf(%4, %15) : (!llvm.ptr, i32) -> i32
    %17 = llvm.load %5 : !llvm.ptr -> i32
    %18 = llvm.add %17, %0  : i32
    llvm.store %18, %5 : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb6:  // pred: ^bb1
    llvm.return
  }
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

// CHECK-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
// CHECK-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-NEXT:   llvm.func @main() attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
// CHECK-NEXT:     %0 = clift.make_label !clift.label
// CHECK-NEXT:     %1 = clift.make_label !clift.label
// CHECK-NEXT:     %2 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:     %3 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:     %4 = llvm.mlir.constant(10 : i32) : i32
// CHECK-NEXT:     %5 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
// CHECK-NEXT:     %6 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-NEXT:     %7 = llvm.alloca %2 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:     %8 = llvm.alloca %2 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:     llvm.store %3, %7 : i32, !llvm.ptr
// CHECK-NEXT:     llvm.store %3, %8 : i32, !llvm.ptr
// CHECK-NEXT:     llvm.br ^bb2
// CHECK-NEXT:   ^bb1:  // pred: ^bb2
// CHECK-NEXT:     clift.assign_label %0 !clift.label
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   ^bb2:  // pred: ^bb0
// CHECK-NEXT:     clift.loop {
// CHECK-NEXT:       llvm.br ^bb1
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %9 = llvm.load %7 : !llvm.ptr -> i32
// CHECK-NEXT:       %10 = llvm.icmp "slt" %9, %4 : i32
// CHECK-NEXT:       llvm.cond_br %10, ^bb3, ^bb5
// CHECK-NEXT:     ^bb2:  // pred: ^bb3
// CHECK-NEXT:       clift.loop {
// CHECK-NEXT:         llvm.br ^bb1
// CHECK-NEXT:       ^bb1:  // pred: ^bb0
// CHECK-NEXT:         %15 = llvm.load %8 : !llvm.ptr -> i32
// CHECK-NEXT:         %16 = llvm.icmp "slt" %15, %4 : i32
// CHECK-NEXT:         llvm.cond_br %16, ^bb2, ^bb3
// CHECK-NEXT:       ^bb2:  // pred: ^bb1
// CHECK-NEXT:         %17 = llvm.load %8 : !llvm.ptr -> i32
// CHECK-NEXT:         %18 = llvm.call @printf(%6, %17) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:         %19 = llvm.load %8 : !llvm.ptr -> i32
// CHECK-NEXT:         %20 = llvm.add %19, %2  : i32
// CHECK-NEXT:         llvm.store %20, %8 : i32, !llvm.ptr
// CHECK-NEXT:         llvm.br ^bb4
// CHECK-NEXT:       ^bb3:  // pred: ^bb1
// CHECK-NEXT:         clift.goto %1 !clift.label
// CHECK-NEXT:       ^bb4:  // pred: ^bb2
// CHECK-NEXT:         "clift.continue"() : () -> ()
// CHECK-NEXT:       } ^bb4
// CHECK-NEXT:     ^bb3:  // pred: ^bb1
// CHECK-NEXT:       llvm.br ^bb2
// CHECK-NEXT:     ^bb4:  // pred: ^bb2
// CHECK-NEXT:       clift.assign_label %1 !clift.label
// CHECK-NEXT:       %11 = llvm.load %7 : !llvm.ptr -> i32
// CHECK-NEXT:       %12 = llvm.call @printf(%6, %11) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:       %13 = llvm.load %7 : !llvm.ptr -> i32
// CHECK-NEXT:       %14 = llvm.add %13, %2  : i32
// CHECK-NEXT:       llvm.store %14, %7 : i32, !llvm.ptr
// CHECK-NEXT:       llvm.br ^bb6
// CHECK-NEXT:     ^bb5:  // pred: ^bb1
// CHECK-NEXT:       clift.goto %0 !clift.label
// CHECK-NEXT:     ^bb6:  // pred: ^bb4
// CHECK-NEXT:       "clift.continue"() : () -> ()
// CHECK-NEXT:     } ^bb1
// CHECK-NEXT:   }
// CHECK-COM:   llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
// CHECK-NEXT: }
