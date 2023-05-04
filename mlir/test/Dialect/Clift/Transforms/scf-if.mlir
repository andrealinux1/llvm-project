// RUN: mlir-opt -restructure-clift %s | FileCheck %s -check-prefix=CHECK-RESTRUCTURE
// RUN: mlir-opt -restructure-clift -comb-clift %s | FileCheck %s -check-prefix=CHECK-COMB

func.func @flipif(%cond: i1) -> i32 {
  %x, %y = scf.if %cond -> (i32, i32) {
    %x_true = arith.constant 10 : i32
    %y_true = arith.constant 20 : i32
    scf.yield %x_true, %y_true : i32, i32
  } else {
    %x_false = arith.constant 30 : i32
    %y_false = arith.constant 40 : i32
    scf.yield %x_false, %y_false : i32, i32
  }
  return %x : i32
}

// CHECK-RESTRUCTURE-LABEL:  func.func @flipif(%arg0: i1) -> i32 {
// CHECK-RESTRUCTURE-NEXT:    %0:2 = scf.if %arg0 -> (i32, i32) {
// CHECK-RESTRUCTURE-NEXT:      %c10_i32 = arith.constant 10 : i32
// CHECK-RESTRUCTURE-NEXT:      %c20_i32 = arith.constant 20 : i32
// CHECK-RESTRUCTURE-NEXT:      scf.yield %c10_i32, %c20_i32 : i32, i32
// CHECK-RESTRUCTURE-NEXT:    } else {
// CHECK-RESTRUCTURE-NEXT:      %c30_i32 = arith.constant 30 : i32
// CHECK-RESTRUCTURE-NEXT:      %c40_i32 = arith.constant 40 : i32
// CHECK-RESTRUCTURE-NEXT:      scf.yield %c30_i32, %c40_i32 : i32, i32
// CHECK-RESTRUCTURE-NEXT:    }
// CHECK-RESTRUCTURE-NEXT:    return %0#0 : i32
// CHECK-RESTRUCTURE-NEXT:  }

// CHECK-COMB-LABEL:   func.func @flipif(%arg0: i1) -> i32 {
// CHECK-COMB-NEXT:     %0:2 = scf.if %arg0 -> (i32, i32) {
// CHECK-COMB-NEXT:       %c10_i32 = arith.constant 10 : i32
// CHECK-COMB-NEXT:       %c20_i32 = arith.constant 20 : i32
// CHECK-COMB-NEXT:       scf.yield %c10_i32, %c20_i32 : i32, i32
// CHECK-COMB-NEXT:     } else {
// CHECK-COMB-NEXT:       %c30_i32 = arith.constant 30 : i32
// CHECK-COMB-NEXT:       %c40_i32 = arith.constant 40 : i32
// CHECK-COMB-NEXT:       scf.yield %c30_i32, %c40_i32 : i32, i32
// CHECK-COMB-NEXT:     }
// CHECK-COMB-NEXT:     return %0#0 : i32
// CHECK-COMB-NEXT:   }
