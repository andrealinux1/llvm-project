#include "mlir/Dialect/Clift/IR/CliftTypes.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"

void mlir::clift::CliftDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Clift/IR/CliftOpsTypes.h.inc"
      >();
}
