#include "mlir/Dialect/Clift/IR/CliftOps.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Clift/IR/CliftOps.cpp.inc"

void f() { mlir::clift::MakeLabelOp op; }
