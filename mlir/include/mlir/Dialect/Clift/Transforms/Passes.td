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

def RestructureClift : Pass<"restructure-clift"> {
  let summary = "Restructure in CLift module";
  let constructor = "mlir::createRestructureCliftPass()";
  let options = [
    Option<"restructure", "perform-restructure", "bool",
           /*default=*/"true",
           "Perform the restructure operation.">
  ];
  let dependentDialects = ["clift::CliftDialect"];
}

def CombClift : Pass<"comb-clift"> {
  let summary = "Comb in Clift module";
  let constructor = "mlir::createCombCliftPass()";
  let options = [
    Option<"comb", "perform-comb", "bool",
           /*default=*/"true",
           "Perform the combing operation.">
  ];
  let dependentDialects = ["clift::CliftDialect"];
}
#endif // MLIR_DIALECT_CLIFT_PASSES
