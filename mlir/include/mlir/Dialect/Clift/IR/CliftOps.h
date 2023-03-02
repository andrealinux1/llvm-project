#pragma once

#include "mlir/Dialect/Clift/IR/Clift.h"
#include "mlir/Dialect/Clift/IR/CliftTraits.h"
#include "mlir/Dialect/Clift/IR/CliftTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Clift/IR/CliftOps.h.inc"

/*static void examples() {*/
/*mlir::clift::AddOp *add;*/
/*mlir::clift::ConstantOp *constant;*/
/*mlir::clift::FunctionOp *function;*/
/*mlir::clift::LoopOp *loop;*/
/*}*/
