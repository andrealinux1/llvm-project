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

#include "mlir/Dialect/Clift/Analysis/InlineEdgeAnalysis.h"
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
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_COMBCLIFT
#include "mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

class CombCliftImpl {
  using EdgeDescriptor = revng::detail::EdgeDescriptor<mlir::Block *>;
  using EdgeSet = llvm::SmallSet<EdgeDescriptor, 4>;

public:
  CombCliftImpl(DominanceInfo &DomInfo, PostDominanceInfo &PostDomInfo)
      : DomInfo(DomInfo), PostDomInfo(PostDomInfo) {}

  void run(mlir::Region &LoopRegion, mlir::PatternRewriter &Rewriter);

private:
  bool updateTerminatorOperands(mlir::Block *B, IRMapping &Mapping);
  llvm::SmallVector<mlir::Block *>
  collectConditionalBlocks(mlir::Region &LoopRegion);
  llvm::SmallVector<mlir::Block *>
  insertDummyDominators(mlir::Region &LoopRegion, mlir::Block *Conditional,
                        CliftInlinedEdge<mlir::Block *> &InlinedEdges,
                        mlir::PatternRewriter &Rewriter);
  mlir::Block *electPostDom(mlir::Block *Conditional);
  void performCombOperation(mlir::Region &LoopRegion,
                            llvm::SmallVector<mlir::Block *> &DummyDominators,
                            mlir::Block *PostDom,
                            mlir::PatternRewriter &Rewriter);

private:
  DominanceInfo &DomInfo;
  PostDominanceInfo &PostDomInfo;
};

bool CombCliftImpl::updateTerminatorOperands(mlir::Block *B,
                                             IRMapping &Mapping) {
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

llvm::SmallVector<mlir::Block *>
CombCliftImpl::collectConditionalBlocks(mlir::Region &LoopRegion) {
  llvm::SmallVector<mlir::Block *> ConditionalBlocks;
  for (mlir::Block *B : llvm::post_order(&(LoopRegion.front()))) {

    // Enqueue all blocks with more than one successor as conditional nodes to
    // process.
    if (successor_range_size(B) >= 2) {
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

  return ConditionalBlocks;
}

llvm::SmallVector<mlir::Block *> CombCliftImpl::insertDummyDominators(
    mlir::Region &LoopRegion, mlir::Block *Conditional,
    CliftInlinedEdge<mlir::Block *> &InlinedEdges,
    mlir::PatternRewriter &Rewriter) {
  // Insert a new dummy node between the conditional block and its immediate
  // successors. In this way, we can unify the handling of conditional and
  // switch blocks, always working on the dominance of this new
  // dummy-frontier block with respect to the blocks belonging to the branch
  // under analysis.
  llvm::SmallVector<mlir::Block *> DummyDominators;
  for (mlir::Block *Successor : successor_range(Conditional)) {

    // Skip over the inlined edges, do not create a `DummyDominator` for it,
    // and do not add it to the nodes that need comb processing.
    if (InlinedEdges.isInlined(EdgeDescriptor(Conditional, Successor))) {
      continue;
    }

    // Create a new empty block, which will point to the original successor.
    mlir::Block *DummyDominator = Rewriter.createBlock(&LoopRegion);
    DummyDominators.push_back(DummyDominator);
    Rewriter.setInsertionPointToEnd(DummyDominator);
    MLIRContext *Context = LoopRegion.getContext();
    auto Loc = UnknownLoc::get(Context);
    Rewriter.create<LLVM::BrOp>(Loc, Successor);

    // Update trees.
    DomInfo.getDomTree(&LoopRegion).insertEdge(DummyDominator, Successor);
    PostDomInfo.getDomTree(&LoopRegion).insertEdge(DummyDominator, Successor);

    // Connect the `Conditional` block to the newly created
    // `DummyDominator`.
    IRMapping IRMapping;
    IRMapping.map(Successor, DummyDominator);
    updateTerminatorOperands(Conditional, IRMapping);

    // Update trees.
    DomInfo.getDomTree(&LoopRegion).insertEdge(Conditional, DummyDominator);
    DomInfo.getDomTree(&LoopRegion).deleteEdge(Conditional, Successor);
    PostDomInfo.getDomTree(&LoopRegion).insertEdge(Conditional, DummyDominator);
    PostDomInfo.getDomTree(&LoopRegion).deleteEdge(Conditional, Successor);
  }

  return DummyDominators;
}

mlir::Block *CombCliftImpl::electPostDom(mlir::Block *Conditional) {
  // Retrieve the post dominator of the conditional node. The post dominator
  // may be a `nullptr`, which signals the fact that we should continue with
  // the analysis until the last nodes of the `LoopRegion`.
  // The post dominator should correctly be computed with respect to the
  // original `Conditional`, and not to the `DummyDominator`, or the visit
  // will wrongly stop earlier.
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

  return PostDom;
}

void CombCliftImpl::performCombOperation(
    mlir::Region &LoopRegion, llvm::SmallVector<mlir::Block *> &DummyDominators,
    mlir::Block *PostDom, mlir::PatternRewriter &Rewriter) {

  // The comb analysis should run on each of the previously inserted
  // `DummyDominators`.
  for (mlir::Block *DummyDominator : DummyDominators) {
    // Instantiate a DFS visit, using the `ext` set in order to stop the
    // visit
    // at the immediate post dominator node. If we cannot find the
    // postdominator node for a specific conditional node, and therefore
    // obtaining a `nullptr` as the post dominator node, this is coherently
    // handled by the DFS visit, which should not stop at any node given its
    // `ext` set is composed by a single `nullptr` node.
    llvm::df_iterator_default_set<mlir::Block *> PostDomSet;
    PostDomSet.insert(PostDom);
    for (mlir::Block *DFSBlock :
         llvm::depth_first_ext(DummyDominator, PostDomSet)) {

      // Debug print.
      llvm::dbgs() << "\nEvaluating node: ";
      printBlock(DFSBlock);

      // For each node encountered during the DFS visit, we evaluate the
      // dominance criterion by its conditional node, and in case it is not
      // dominated by the conditional, we need to perform the comb
      // operation.
      if (not DomInfo.dominates(DummyDominator, DFSBlock)) {

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

        // Incremental update of the dominator and post dominator trees to
        // encompass the newly created cloned node.
        // TODO: since the block that we are adding to the trees are not yet
        // attached to incoming node, so it is not possible to specify the
        // actual dominator (as the `addNewBlock` method would like). I'm
        // passing `nullptr` for now, but suspect this will break.
        // DomInfo.getDomTree(&LoopRegion).addNewBlock(DFSBlockClone,
        // nullptr);
        // PostDomInfo.getDomTree(&LoopRegion).addNewBlock(DFSBlockClone,
        // nullptr);

        // Incremental update of the dominator and post dominator trees to
        // represent the exiting edges of the `DFSBlockClone` which are
        // identical to the ones of `DFSBlock`.
        for (mlir::Block *Successor : successor_range(DFSBlock)) {
          DomInfo.getDomTree(&LoopRegion).insertEdge(DFSBlockClone, Successor);
          PostDomInfo.getDomTree(&LoopRegion)
              .insertEdge(DFSBlockClone, Successor);
        }

        // Adjust the predecessors of the combed node, so that:
        // - The predecessors that are are dominated by the conditional
        // node, still point to `DFSBlock`.
        // - The predecessors that do not satisfy this condition, are
        // modified in order to point to the newly created `DFSBlockClone`.
        llvm::SmallVector<mlir::Block *> NotDominatedPredecessors;
        for (mlir::Block *Predecessor : predecessor_range(DFSBlock)) {
          if (not DomInfo.dominates(DummyDominator, Predecessor)) {
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

          // Perform the incremental update of the dominator and post
          // dominator trees accordingly to the CFG modification.
          DomInfo.getDomTree(&LoopRegion)
              .insertEdge(Predecessor, DFSBlockClone);
          DomInfo.getDomTree(&LoopRegion).deleteEdge(Predecessor, DFSBlock);
          PostDomInfo.getDomTree(&LoopRegion)
              .insertEdge(Predecessor, DFSBlockClone);
          PostDomInfo.getDomTree(&LoopRegion).deleteEdge(Predecessor, DFSBlock);
        }

        // We should verify that at least one of the predecessor has been
        // adjusted using
        assert(Updated);
      }
    }
  }
}

void CombCliftImpl::run(mlir::Region &LoopRegion,
                        mlir::PatternRewriter &Rewriter) {
  // TODO: implement implementation here.

  // Perform the `InlineEdgeAnalysis` over the current region.
  CliftInlinedEdge<mlir::Block *> InlinedEdges(LoopRegion, DomInfo,
                                               PostDomInfo);

  // Collect all the conditional nodes in the loop region.
  llvm::SmallVector<mlir::Block *> ConditionalBlocks =
      collectConditionalBlocks(LoopRegion);

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

    // Insert the dummy dominators blocks for each successor of the conditional
    // block.
    llvm::SmallVector<mlir::Block *> DummyDominators =
        insertDummyDominators(LoopRegion, Conditional, InlinedEdges, Rewriter);

    // Elect the immediate post dominator for the current `Conditional` block.
    mlir::Block *PostDom = electPostDom(Conditional);

    // Perform the comb operation.
    performCombOperation(LoopRegion, DummyDominators, PostDom, Rewriter);

    // Debug print to make the module serialization go on new line.
    llvm::dbgs() << "\n";
  }
}

class MatchAndRewriteImpl {
public:
  MatchAndRewriteImpl() {}

  mlir::LogicalResult run(Operation *Op, mlir::PatternRewriter &Rewriter,
                          DominanceInfo &DomInfo,
                          PostDominanceInfo &PostDomInfo);
};

mlir::LogicalResult MatchAndRewriteImpl::run(Operation *Op,
                                             mlir::PatternRewriter &Rewriter,
                                             DominanceInfo &DomInfo,
                                             PostDominanceInfo &PostDomInfo) {
  // Ensure that each `LLVM.LLVMFuncOp` operation that we find, contains a
  // single region that we apply combing to.
  assert(Op->getNumRegions() == 1);

  // Before calling the region transformation operation, we should ensure that
  // we do not find an empty `LLVM.LLVMFuncOp` operation.
  mlir::Region &LoopRegion = Op->getRegion(0);
  assert(not LoopRegion.getBlocks().empty());

  // We also check that the `LLVM.LLVMFuncOp` region is a DAG.
  // TODO: this assertion should eventually be integrated by the verifier on
  // `clift.loop` operations.
  assert(isDAG(&LoopRegion));

  llvm::dbgs() << "\nPerforming comb on operation:\n";
  Op->dump();

  // Instantiate the `CombCliftImpl` class and call the `run` method to
  // perform the actual comb operation.
  // This additional class is needed because we need two `OpRewritePattern`
  // classes, one for `clift::LoopOp` and one for the root region contained in
  // the `LLVM::LLVMFuncOp`.
  CombCliftImpl CCI(DomInfo, PostDomInfo);
  CCI.run(LoopRegion, Rewriter);

  return success();
}

class FuncOpRewriter : public OpRewritePattern<LLVM::LLVMFuncOp> {
  DominanceInfo &DomInfo;
  PostDominanceInfo &PostDomInfo;

public:
  FuncOpRewriter(MLIRContext *Context, DominanceInfo &DomInfo,
                 PostDominanceInfo &PostDomInfo)
      : OpRewritePattern<LLVM::LLVMFuncOp>(Context), DomInfo(DomInfo),
        PostDomInfo(PostDomInfo) {}

  mlir::LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp Op,
                  mlir::PatternRewriter &Rewriter) const final {
    MatchAndRewriteImpl MRI;
    return MRI.run(Op, Rewriter, DomInfo, PostDomInfo);
  }
};

class CliftLoopRewriter : public OpRewritePattern<clift::LoopOp> {
  DominanceInfo &DomInfo;
  PostDominanceInfo &PostDomInfo;

public:
  CliftLoopRewriter(MLIRContext *Context, DominanceInfo &DomInfo,
                    PostDominanceInfo &PostDomInfo)
      : OpRewritePattern<clift::LoopOp>(Context), DomInfo(DomInfo),
        PostDomInfo(PostDomInfo) {}

  mlir::LogicalResult
  matchAndRewrite(clift::LoopOp Op,
                  mlir::PatternRewriter &Rewriter) const final {
    MatchAndRewriteImpl MRI;
    return MRI.run(Op, Rewriter, DomInfo, PostDomInfo);
  }
};

struct CombClift : public impl::CombCliftBase<CombClift> {
  void runOnOperation() override {
    DominanceInfo &DomInfo = getAnalysis<DominanceInfo>();
    PostDominanceInfo &PostDomInfo = getAnalysis<PostDominanceInfo>();

    RewritePatternSet Patterns(&getContext());
    Patterns.add<FuncOpRewriter>(&getContext(), DomInfo, PostDomInfo);
    Patterns.add<CliftLoopRewriter>(&getContext(), DomInfo, PostDomInfo);

    // Check that the root region of the function is a DAG. We need this
    // explicit check because the root region is not encapsuled in a
    // `clift.loop` operation.
    SmallVector<Operation *> Functions;
    getOperation()->walk([&](LLVM::LLVMFuncOp F) {
      mlir::Region &FunctionRegion = F->getRegion(0);
      assert(F->getNumRegions() == 1);
      if (not FunctionRegion.getBlocks().empty()) {
        Functions.push_back(F);
      }
    });

    // Verify that the body of each function is a DAG.
    for (Operation *F : Functions) {
      mlir::Region &FunctionRegion = F->getRegion(0);

      // TODO: this assertion should eventually be integrated by the verifier on
      // `clift.loop` operations.
      assert(isDAG(&FunctionRegion));
    }

    SmallVector<Operation *> CliftLoops;
    getOperation()->walk([&](clift::LoopOp L) { CliftLoops.push_back(L); });

    // Reverse the order computed by the `walk` visit on the `clift.loop`
    // operations, which is by default a `PostOrder` visit, so that we end up
    // with the reverse post order. This is necessary because, the subsequent
    // `applyOpPatternsAndFold` method, processes the worklist in a reverse
    // fashion, since it uses the `pop_back` method to extract elements from the
    // `WorkList` container. Our goal is to process the operations in post
    // order.
    std::reverse(CliftLoops.begin(), CliftLoops.end());

    // Accumulate both the `LLVM.LLVMFuncOp` and the `clift.loop` operations.
    SmallVector<Operation *> WorkList;
    WorkList.insert(WorkList.end(), Functions.begin(), Functions.end());
    WorkList.insert(WorkList.end(), CliftLoops.begin(), CliftLoops.end());

    auto Strictness = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(
            applyOpPatternsAndFold(WorkList, std::move(Patterns), Strictness)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createCombCliftPass() {
  return std::make_unique<CombClift>();
}
