// RUN: mlir-opt -restructure-clift %s | FileCheck %s

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

// CHECK:  func.func @flipif(%arg0: i1) -> i32 {
// CHECK-NEXT:    %0:2 = scf.if %arg0 -> (i32, i32) {
// CHECK-NEXT:      %c10_i32 = arith.constant 10 : i32
// CHECK-NEXT:      %c20_i32 = arith.constant 20 : i32
// CHECK-NEXT:      scf.yield %c10_i32, %c20_i32 : i32, i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %c30_i32 = arith.constant 30 : i32
// CHECK-NEXT:      %c40_i32 = arith.constant 40 : i32
// CHECK-NEXT:      scf.yield %c30_i32, %c40_i32 : i32, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#0 : i32
// CHECK-NEXT:  }
