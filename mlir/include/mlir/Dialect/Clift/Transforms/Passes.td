//===-- Passes.td - Clift pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CLIFT_PASSES
#define MLIR_DIALECT_CLIFT_PASSES

include "mlir/Pass/PassBase.td"

def RestructureClift
    : Pass<"restructure-clift"> {
  let summary = "Restructure in CLift module";
  let constructor = "mlir::createRestructureCliftPass()";
  let dependentDialects = [];
}

#endif // MLIR_DIALECT_CLIFT_PASSES
