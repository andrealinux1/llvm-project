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
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_COMBCLIFT
#include "mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

class CombCliftRewriter : public OpRewritePattern<clift::LoopOp> {

  DominanceInfo &DomInfo;
  PostDominanceInfo &PostDomInfo;

public:
  CombCliftRewriter(MLIRContext *Context, DominanceInfo &DomInfo,
                    PostDominanceInfo &PostDomInfo)
      : OpRewritePattern<clift::LoopOp>(Context), DomInfo(DomInfo),
        PostDomInfo(PostDomInfo) {}

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

  bool updateTerminatorOperands(mlir::Block *B, IRMapping &Mapping) const {
    bool UpdatedOperand = false;
    Operation *Terminator = B->getTerminator();
    for (auto &SuccOp : Terminator->getBlockOperands()) {
      if (mlir::Block *MappedBlock = Mapping.lookupOrNull(SuccOp.get())) {
        SuccOp.set(MappedBlock);
        UpdatedOperand = true;
      }
    }

    return UpdatedOperand;
  }

  void performCombCliftRegion(mlir::Region &LoopRegion,
                              mlir::PatternRewriter &Rewriter) const {

    // TODO: implement implementation here.

    // Step 1: Collect all the conditional nodes in the loop region.
    assert(not LoopRegion.empty());
    llvm::SmallVector<mlir::Block *> ConditionalBlocks;
    for (mlir::Block *B : llvm::post_order(&(LoopRegion.front()))) {

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

    // Iterate over the collected conditional nodes, and over the nodes between
    // the conditional and its immediate postdominator.
    // We need to inspect the collected conditional blocks in reverse reverse
    // post order.
    while (not ConditionalBlocks.empty()) {
      mlir::Block *Conditional = ConditionalBlocks.back();
      ConditionalBlocks.pop_back();

      // Debug print.
      llvm::dbgs() << "\nEvaluating conditional node: ";
      printBlock(Conditional);

      // Retrieve the post dominator of the confitional node. The post dominator
      // may be a `nullptr`, which signals the fact that we should continue with
      // the analysis until the last nodes of the `LoopRegion`.
      auto *PostDomNode = PostDomInfo.getNode(Conditional);
      mlir::Block *PostDom = PostDomNode->getIDom()->getBlock();

      // Debug print of the identified immediate post dominator for the
      // conditional node under analysis.
      llvm::dbgs() << "\nIdentified post dominator is : ";
      if (PostDom != nullptr) {
        printBlock(PostDom);
      } else {
        llvm::dbgs() << "nullptr";
      }
      llvm::dbgs() << "\n";

      // Instantiate a DFS visit, using the `ext` set in order to stop the visit
      // at the immediate post dominator node. If we cannot find the
      // postdominator node for a specific conditional node, and therefore
      // obtaining a `nullptr` as the post dominator node, this is coherently
      // handled by the DFS visit, which should not stop at any node given its
      // `ext` set is composed by a single `nullptr` node.
      llvm::df_iterator_default_set<mlir::Block *> PostDomSet;
      PostDomSet.insert(PostDom);
      for (mlir::Block *DFSBlock :
           llvm::depth_first_ext(Conditional, PostDomSet)) {

        // Debug print.
        llvm::dbgs() << "\nEvaluating node: ";
        printBlock(DFSBlock);

        // For each node encountered during the DFS visit, we evaluate the
        // dominance criterion by its conditional node, and in case it is not
        // dominated by the conditional, we need to perform the comb operation.
        if (not DomInfo.dominates(Conditional, DFSBlock)) {

          // Debug print.
          llvm::dbgs() << "\nNode is not dominated, comb";

          // We perform here the combing of the `DFSNode` identified as not
          // dominated by the conditional node it is reachable from.

          // Manual cloning of the block body.
          IRMapping Mapping;
          mlir::Block *DFSBlockClone = Rewriter.createBlock(DFSBlock);
          Mapping.map(DFSBlock, DFSBlockClone);

          // Iterate over all the operations contained in the `DFSBlock`, and
          // clone them.
          for (auto &BlockOp : *DFSBlock) {
            Operation *CloneOp = Rewriter.clone(BlockOp, Mapping);
            Mapping.map(BlockOp.getResults(), CloneOp->getResults());
          }

          // Adjust the predecessors of the combed node, so that:
          // - The predecessors that are are dominated by the conditional node,
          // still point to `DFSBlock`.
          // - The predecessors that do not satisfy this condition, are modified
          // in order to point to the newly created `DFSBlockClone`.
          llvm::SmallVector<mlir::Block *> NotDominatedPredecessors;
          for (mlir::Block *Predecessor : predecessor_range(DFSBlock)) {
            if (not DomInfo.dominates(Conditional, Predecessor)) {
              NotDominatedPredecessors.push_back(Predecessor);

              // Debug print.
              llvm::dbgs() << "\nPredecessor causing not dominance: ";
              printBlock(Predecessor);
            }
          }

          // Map the predecessor termnators's targets to the newly created
          // `DFSBlockClone`.
          bool Updated = false;
          for (mlir::Block *Predecessor : NotDominatedPredecessors) {
            Updated |= updateTerminatorOperands(Predecessor, Mapping);
          }

          // We should verify that at least one of the predecessor has been
          // adjusted using
          assert(Updated);
        }
      }

      // Debug print to make the module serialization go on new line.
      llvm::dbgs() << "\n";
    }
  }
};

struct CombClift : public impl::CombCliftBase<CombClift> {
  void runOnOperation() override {
    DominanceInfo &DomInfo = getAnalysis<DominanceInfo>();
    PostDominanceInfo &PostDomInfo = getAnalysis<PostDominanceInfo>();

    RewritePatternSet Patterns(&getContext());
    Patterns.add<CombCliftRewriter>(&getContext(), DomInfo, PostDomInfo);

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
