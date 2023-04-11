//===- CombCliftPass.cpp - perform restructuring on Clift -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Performs comb on Clift.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Clift/Transforms/Passes.h"

#include "mlir/Dialect/Clift/IR/Clift.h"
#include "mlir/Dialect/Clift/IR/CliftOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_COMBCLIFT
#include "mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

class CombCliftRewriter : public OpRewritePattern<clift::LoopOp> {

  using mlir::OpRewritePattern<clift::LoopOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(clift::LoopOp Op,
                  mlir::PatternRewriter &Rewriter) const final {

    // Ensure that each `clift.loop` operation that we find, contains a single
    // region that we apply combing to.
    assert(Op->getNumRegions() == 1);

    // Before calling the region transformation operation, we should ensure that
    // we do not find an empty `clift.loop` operation.
    mlir::Region &LoopRegion = Op->getRegion(0);
    assert(not LoopRegion.getBlocks().empty());
    performCombCliftRegion(LoopRegion, Rewriter);

    return success();
  }

  void performCombCliftRegion(mlir::Region &FunctionRegion,
                              mlir::PatternRewriter &Rewriter) const {

    // TODO: implement implementation here.
  }
};

struct CombClift : public impl::CombCliftBase<CombClift> {
  void runOnOperation() override {
    RewritePatternSet Patterns(&getContext());
    Patterns.add<CombCliftRewriter>(&getContext());

    SmallVector<Operation *> CliftLoops;
    getOperation()->walk([&](clift::LoopOp L) { CliftLoops.push_back(L); });

    auto Strictness = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyOpPatternsAndFold(CliftLoops, std::move(Patterns),
                                      Strictness)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createCombCliftPass() {
  return std::make_unique<CombClift>();
}
