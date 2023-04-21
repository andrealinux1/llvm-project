#include "mlir/Dialect/Clift/IR/CliftOps.h"

#include "mlir/GraphAlgorithms/GraphAlgorithms.h"
#include "mlir/IR/RegionGraphTraits.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Clift/IR/CliftOps.cpp.inc"

void mlir::clift::CliftDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Clift/IR/CliftOps.cpp.inc"
      >();
}

//===-----------------------------------------------------------------========//
// Code for clift::AssignLabelOp.
//===----------------------------------------------------------------------===//

mlir::LogicalResult
mlir::clift::AssignLabelOp::canonicalize(mlir::clift::AssignLabelOp op,
                                         mlir::PatternRewriter &rewriter) {
  for (const mlir::OpOperand &use : op.getLabel().getUses())
    if (mlir::isa<mlir::clift::GoToOp>(use.getOwner()))
      return mlir::success();

  rewriter.eraseOp(op);
  return mlir::success();
}

//===-----------------------------------------------------------------========//
// Code for clift::MakeLabelOp.
//===----------------------------------------------------------------------===//

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

//===-----------------------------------------------------------------========//
// Code for clift::LoopOp.
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::clift::LoopOp::verifyRegions() {

  // Verify that the region inside each `clift.loop` is acyclic.

  // TODO: this verifier does not cover the root region, because that is not
  //       inside a `clift.loop`. A solution to this may be the definition of a
  //       `IsCombableInterface` in order to perform the verification on the
  //       interface and not on the operation.
  mlir::Region &LoopOpRegion = getBody();
  if (isDAG(&LoopOpRegion)) {
    return success();
  } else {
    return failure();
  }
}
