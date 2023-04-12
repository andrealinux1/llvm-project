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
#include "mlir/Dialect/Clift/IR/CliftDebug.h"
#include "mlir/Dialect/Clift/IR/CliftOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/GraphAlgorithms/GraphAlgorithms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DepthFirstIterator.h"
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

    llvm::dbgs() << "\nPerforming comb on operation:\n";
    Op->dump();
    performCombCliftRegion(LoopRegion, Rewriter);

    return success();
  }

  void performCombCliftRegion(mlir::Region &LoopRegion,
                              mlir::PatternRewriter &Rewriter) const {

    // TODO: implement implementation here.

    // Step 1: Collect all the conditional nodes in the loop region.
    assert(not LoopRegion.empty());
    llvm::SmallVector<mlir::Block *> ConditionalBlocks;
    for (mlir::Block *B : llvm::depth_first(&(LoopRegion.front()))) {

      // TODO: current implementation is collecting conditional nodes with two
      // successors, and asserting if they are greater. Handle also switch
      // nodes.
      if (successor_range_size(B) >= 2) {
        assert(successor_range_size(B) == 2);
        ConditionalBlocks.push_back(B);
      }
    }

    // Print as debug the collected conditional nodes for comb.
    llvm::dbgs() << "\nConditional Nodes: ";
    for (mlir::Block *B : ConditionalBlocks) {
      llvm::dbgs() << "\n";
      printBlock(B);
    }
    llvm::dbgs() << "\n";
  }
};

struct CombClift : public impl::CombCliftBase<CombClift> {
  void runOnOperation() override {
    RewritePatternSet Patterns(&getContext());
    Patterns.add<CombCliftRewriter>(&getContext());

    SmallVector<Operation *> CliftLoops;

    // TODO: the `walk` function contained in `include/mlir/IR/Visitors.h`,
    // should accept as an additional parameter an ordering indication about
    // which visit to follow during the walk on the operations.
    // getOperation()->walk([&](clift::LoopOp L) { CliftLoops.push_back(L); },
    // WalkOrder::PostOrder);
    // mlir::detail::walk(getOperation(), [&](clift::LoopOp L) {
    // CliftLoops.push_back(L); }, WalkOrder::PostOrder);

    // TODO: improve this behavior with the correct use of the
    // `applyPatternsAndFoldGreedily`.
    getOperation()->walk([&](clift::LoopOp L) { CliftLoops.push_back(L); });

    // TODO: Reverse the order computed by the `walk` visit, which is by default
    // a `PostOrder` visit, so that we end up with the reverse post order. For
    // some reason, the subsequent `applyOpPatternsAndFold` method process the
    // worklist in a reverse fashion, so we need an additional reverse operation
    // in the middle, not completely clear why.
    std::reverse(CliftLoops.begin(), CliftLoops.end());

    auto Strictness = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyOpPatternsAndFold(CliftLoops, std::move(Patterns),
                                      Strictness)))
      signalPassFailure();

    /*
    GreedyRewriteConfig Grc;
    Grc.useTopDownTraversal = false;
    Grc.maxIterations = 5;

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(Patterns),
    Grc))) signalPassFailure();
    */
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createCombCliftPass() {
  return std::make_unique<CombClift>();
}
