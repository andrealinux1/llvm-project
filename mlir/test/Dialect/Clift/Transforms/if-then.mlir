// RUN: mlir-opt -restructure-clift %s | FileCheck %s -check-prefix=CHECK-RESTRUCTURE
// RUN: mlir-opt -restructure-clift -comb-clift %s | FileCheck %s -check-prefix=CHECK-COMB

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
  llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @main() attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.addressof @".str" : !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %5 : i32, !llvm.ptr
    %6 = llvm.load %5 : !llvm.ptr -> i32
    %7 = llvm.icmp "ne" %6, %1 : i32
    llvm.cond_br %7, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.store %0, %5 : i32, !llvm.ptr
    llvm.br ^bb3
  ^bb2:  // pred: ^bb0
    llvm.store %2, %5 : i32, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %8 = llvm.load %5 : !llvm.ptr -> i32
    %9 = llvm.call @printf(%4, %8) : (!llvm.ptr, i32) -> i32
    llvm.return
  }
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

// CHECK-RESTRUCTURE-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
// CHECK-RESTRUCTURE-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-RESTRUCTURE-NEXT:   llvm.func @main() attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
// CHECK-RESTRUCTURE-NEXT:     %0 = llvm.mlir.constant(1 : i32) : i32
// CHECK-RESTRUCTURE-NEXT:     %1 = llvm.mlir.constant(0 : i32) : i32
// CHECK-RESTRUCTURE-NEXT:     %2 = llvm.mlir.constant(2 : i32) : i32
// CHECK-RESTRUCTURE-NEXT:     %3 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
// CHECK-RESTRUCTURE-NEXT:     %4 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.store %1, %5 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     %6 = llvm.load %5 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %7 = llvm.icmp "ne" %6, %1 : i32
// CHECK-RESTRUCTURE-NEXT:     llvm.cond_br %7, ^bb1, ^bb2
// CHECK-RESTRUCTURE-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-RESTRUCTURE-NEXT:     llvm.store %0, %5 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb3
// CHECK-RESTRUCTURE-NEXT:   ^bb2:  // pred: ^bb0
// CHECK-RESTRUCTURE-NEXT:     llvm.store %2, %5 : i32, !llvm.ptr
// CHECK-RESTRUCTURE-NEXT:     llvm.br ^bb3
// CHECK-RESTRUCTURE-NEXT:   ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK-RESTRUCTURE-NEXT:     %8 = llvm.load %5 : !llvm.ptr -> i32
// CHECK-RESTRUCTURE-NEXT:     %9 = llvm.call @printf(%4, %8) : (!llvm.ptr, i32) -> i32
// CHECK-RESTRUCTURE-NEXT:     llvm.return
// CHECK-RESTRUCTURE-NEXT:   }
// CHECK-RESTRUCTURE-COM:   llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
// CHECK-RESTRUCTURE-NEXT: }

// CHECK-COMB-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
// CHECK-COMB-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-COMB-NEXT:   llvm.func @main() attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
// CHECK-COMB-NEXT:     %0 = llvm.mlir.constant(1 : i32) : i32
// CHECK-COMB-NEXT:     %1 = llvm.mlir.constant(0 : i32) : i32
// CHECK-COMB-NEXT:     %2 = llvm.mlir.constant(2 : i32) : i32
// CHECK-COMB-NEXT:     %3 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
// CHECK-COMB-NEXT:     %4 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-COMB-NEXT:     %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-COMB-NEXT:     llvm.store %1, %5 : i32, !llvm.ptr
// CHECK-COMB-NEXT:     %6 = llvm.load %5 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %7 = llvm.icmp "ne" %6, %1 : i32
// CHECK-COMB-NEXT:     llvm.cond_br %7, ^bb4, ^bb5
// CHECK-COMB-NEXT:   ^bb1:  // pred: ^bb4
// CHECK-COMB-NEXT:     llvm.store %0, %5 : i32, !llvm.ptr
// CHECK-COMB-NEXT:     llvm.br ^bb3
// CHECK-COMB-NEXT:   ^bb2:  // pred: ^bb5
// CHECK-COMB-NEXT:     llvm.store %2, %5 : i32, !llvm.ptr
// CHECK-COMB-NEXT:     llvm.br ^bb3
// CHECK-COMB-NEXT:   ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK-COMB-NEXT:     %8 = llvm.load %5 : !llvm.ptr -> i32
// CHECK-COMB-NEXT:     %9 = llvm.call @printf(%4, %8) : (!llvm.ptr, i32) -> i32
// CHECK-COMB-NEXT:     llvm.return
// CHECK-COMB-NEXT:   ^bb4:  // pred: ^bb0
// CHECK-COMB-NEXT:     llvm.br ^bb1
// CHECK-COMB-NEXT:   ^bb5:  // pred: ^bb0
// CHECK-COMB-NEXT:     llvm.br ^bb2
// CHECK-COMB-NEXT:   }
// CHECK-COMB-COM:   llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
// CHECK-COMB-NEXT: }
