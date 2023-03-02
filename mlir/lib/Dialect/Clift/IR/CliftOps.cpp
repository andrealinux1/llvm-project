#include "mlir/Dialect/Clift/IR/CliftOps.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Clift/IR/CliftOps.cpp.inc"

void mlir::clift::CliftDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Clift/IR/CliftOps.cpp.inc"
      >();
}

mlir::LogicalResult
mlir::clift::AssignLabelOp::canonicalize(mlir::clift::AssignLabelOp op,
                                         mlir::PatternRewriter &rewriter) {
  for (const mlir::OpOperand &use : op.getLabel().getUses())
    if (mlir::isa<mlir::clift::GoToOp>(use.getOwner()))
      return mlir::success();

  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult mlir::clift::MakeLabelOp::verify() {

  const bool hasAssign =
      llvm::any_of(getResult().getUses(), [](mlir::OpOperand &operand) {
        return mlir::isa<mlir::clift::AssignLabelOp>(operand.getOwner());
      });

  if (hasAssign)
    return mlir::success();

  const bool hasGoTo =
      llvm::any_of(getResult().getUses(), [](mlir::OpOperand &operand) {
        return mlir::isa<mlir::clift::GoToOp>(operand.getOwner());
      });

  if (not hasGoTo)
    return mlir::success();

  emitOpError(getOperationName() + " with a " +
              mlir::clift::GoToOp::getOperationName() + " use must have a " +
              mlir::clift::AssignLabelOp::getOperationName() + " use too.");
  return mlir::failure();
}
