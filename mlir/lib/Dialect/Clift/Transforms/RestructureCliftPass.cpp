//===- RestructureCliftPass.cpp - perform restructuring on Clift ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Performs restructuring on Clift.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Clift/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Clift/IR/Clift.h"
#include "mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "mlir/Dialect/Clift/IR/CliftOps.h"
#include "mlir/Dialect/Clift/IR/CliftTypes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/GraphAlgorithms/GraphAlgorithms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_RESTRUCTURECLIFT
#include "mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

class RestructureCliftRewriter : public OpRewritePattern<LLVM::LLVMFuncOp> {
  using EdgeDescriptor = revng::detail::EdgeDescriptor<mlir::Block *>;
  using EdgeSet = llvm::SmallSet<EdgeDescriptor, 4>;
  using BlockSet = llvm::SmallPtrSet<mlir::Block *, 4>;
  using BlockSetVect = llvm::SmallVector<BlockSet, 4>;
  using BlockIntMap = llvm::DenseMap<mlir::Block *, size_t>;
  using BlockVect = llvm::SmallVector<mlir::Block *, 4>;

  using mlir::OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp op,
                  mlir::PatternRewriter &rewriter) const final {
    // llvm::dbgs() << "Invoking matchAndRewrite on " << "\n";
    // op->dump();

    // Ensure that we start from a `LLVMFuncOp` with a single `cf` region.
    assert(op->getNumRegions() == 1);

    // Transform only regions which have an actual size.
    mlir::Region &reg = op->getRegion(0);
    if (not reg.getBlocks().empty()) {
      performRestructureCliftRegion(reg, rewriter);
    }
    return success();
  }

  void printBackedge(EdgeDescriptor &backedge) const {
    llvm::dbgs() << "Backedge: ";
    backedge.first->printAsOperand(llvm::dbgs());
    llvm::dbgs() << " -> ";
    backedge.second->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  void printBackedges(EdgeSet &backedges) const {
    for (EdgeDescriptor backedge : backedges) {
      printBackedge(backedge);
    }
  }

  void printReachableBlocks(EdgeSet &backedges) const {
    for (EdgeDescriptor backedge : backedges) {
      printBackedge(backedge);
      llvm::dbgs() << "We can reach blocks:\n";
      for (mlir::Block *reachable :
           nodesBetween(backedge.second, backedge.first)) {
        reachable->printAsOperand(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
    }
  }

  void printRegions(BlockSetVect &regions) const {
    size_t regionindex = 0;
    for (BlockSet &region : regions) {
      llvm::dbgs() << "Region idx: " << regionindex << " composed by nodes: \n";
      for (mlir::Block *block : region) {
        block->printAsOperand(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
      regionindex++;
    }
  }

  void printMap(llvm::DenseMap<mlir::Block *, size_t> &Map) const {
    llvm::dbgs() << "Map content:\n";
    for (auto const &[K, V] : Map) {
      K->printAsOperand(llvm::dbgs());
      llvm::dbgs() << " -> " << V << "\n";
    }
  }

  void printVector(llvm::SmallVectorImpl<mlir::Block *> &Vector) const {
    for (mlir::Block *Element : Vector) {
      Element->printAsOperand(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  }

  void printPairVector(
      llvm::SmallVectorImpl<std::pair<mlir::Block *, mlir::Block *>> &Vector)
      const {
    for (auto const &[First, Second] : Vector) {
      First->printAsOperand(llvm::dbgs());
      llvm::dbgs() << " -> ";
      Second->printAsOperand(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
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

  // Returns true if during this iteration we outlined a loop construct.
  bool outlineFirstIteration(
      llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>, 4>
          &LateEntryPairs,
      BlockSet &Region, BlockSet &OutlinedNodes, mlir::Block *Entry,
      mlir::PatternRewriter &rewriter) const {
    bool OutlinedCycle = false;

    // For each abnormal entry found, clone the node target of the abnormal
    // entry, and attach it to the previous predecessor.
    IRMapping CloneMapping;
    while (not LateEntryPairs.empty()) {
      auto const &[Predecessor, LateEntry] = LateEntryPairs.back();
      LateEntryPairs.pop_back();

      // I have to manually perform the cloning of the internal of the block
      // body.
      IRMapping Mapping;
      mlir::Block *LateEntryClone = rewriter.createBlock(LateEntry);
      Mapping.map(LateEntry, LateEntryClone);
      CloneMapping.map(LateEntry, Predecessor);

      // Add the outlined nodes to the set that we will return.
      OutlinedNodes.insert(LateEntryClone);

      for (auto &BlockOp : *LateEntry) {
        Operation *Clone = rewriter.clone(BlockOp, Mapping);
        Mapping.map(BlockOp.getResults(), Clone->getResults());
      }

      // Remap block successors that have been already clone on the respective
      // clone (Otherwise we could end up in outlining loops).
      OutlinedCycle |= updateTerminatorOperands(LateEntryClone, CloneMapping);

      // In the predecessor point to the new cloned block.
      updateTerminatorOperands(Predecessor, Mapping);

      // Enqueue the successors. Note that if the successor is the elected
      // entry, do not clone it, because it is correct to jump there. If the
      // successor is outside of the current set region, do not clone it
      // either, this path will be represented with `goto`s at the current
      // stage.
      for (mlir::Block *Successor : LateEntryClone->getSuccessors()) {
        if (setContains(Region, Successor) && Successor != Entry) {
          LateEntryPairs.push_back({LateEntryClone, Successor});
        }
      }
    }

    return OutlinedCycle;
  }

  void updateParent(revng::detail::ParentTree<mlir::Block *> &Pt,
                    BlockSet &Region, BlockSet &OutlinedNodes) const {

    // Re-add all the nodes of the current region to the parent region.
    if (Pt.hasParent(Region)) {

      // Add to the parent region the outlined nodes from the current
      // region, which morally belong to the containing one.
      Pt.getParent(Region).insert(OutlinedNodes.begin(), OutlinedNodes.end());

      // Update the parent region with the currently analyzed region. This
      // step is actually needed to propagate the outlined nodes in the
      // parent of our parent, if present, otherwise it has no effects.
      Pt.getParent(Region).insert(Region.begin(), Region.end());
    }
  }

  void performRegionIdentification(
      mlir::Region &Reg, mlir::PatternRewriter &Rewriter,
      revng::detail::ParentTree<mlir::Block *> &Pt) const {
    // Region identification and first iteration outlining in a fixed point
    // fashion, until first iteration outlining does not generate new cyclic
    // regions.
    bool OutlinedCycle = true;
    while (OutlinedCycle) {
      OutlinedCycle = false;

      // Clear the `ParentTree` object, a new iteration of the region
      // identification process is run.
      Pt.clear();

      EdgeSet backedges = getBackedges(&Reg.front());
      llvm::dbgs() << "\nInitial backedges:\n";
      printBackedges(backedges);
      llvm::dbgs() << "\nInitial reachables:\n";
      printReachableBlocks(backedges);

      // Declare a metaobject which will contain all the identified region
      // objects.
      BlockSetVect regions;

      // Push manually the `root` region in the identified regions.
      BlockSet RootRegion;
      for (mlir::Block &B : Reg) {
        RootRegion.insert(&B);
      }
      regions.push_back(RootRegion);

      // Collect all the identified regions in a single vector.
      for (EdgeDescriptor backedge : backedges) {
        BlockSet regionNodes = nodesBetween(backedge.second, backedge.first);
        regions.push_back(std::move(regionNodes));
      }

      llvm::dbgs() << "\nInitial regions:\n";
      printRegions(regions);

      // Simplify the identified regions in order to satisfy the requirements.
      simplifyRegions(regions);
      llvm::dbgs() << "\nAfter simplification:\n";
      printRegions(regions);

      // Order the identified regions.
      sortRegions(regions);
      llvm::dbgs() << "\nAfter sorting:\n";
      printRegions(regions);

      // Insert the regions in the region tree.
      for (BlockSet &Region : regions) {
        Pt.insertRegion(Region);
      }

      // Order the regions inside the `ParentTree`. This invokes region
      // reordering
      Pt.order();

      // Order the regions so that they go from the outer one to the inner one.
      llvm::dbgs() << "\nAfter ordering:\n";
      printRegions(Pt.getRegions());

      // Compute the Reverse Post Order.
      llvm::SmallVector<mlir::Block *, 4> RPOT;
      using RPOTraversal = llvm::ReversePostOrderTraversal<mlir::Region *>;
      llvm::copy(RPOTraversal(&Reg), std::back_inserter(RPOT));

      // Compute the distance of each node from the entry node.
      llvm::DenseMap<mlir::Block *, size_t> ShortestPathFromEntry =
          computeDistanceFromEntry(&Reg);

      size_t RegionIndex = 0;
      for (BlockSet &region : Pt.regions()) {
        llvm::dbgs() << "\nRestructuring region idx: " << RegionIndex << ":\n";
        llvm::DenseMap<mlir::Block *, size_t> EntryCandidates =
            getEntryCandidates<mlir::Block *>(region);

        // In case we are analyzing the root region, we expect to have no entry
        // candidates.
        bool RootRegionIteration = (RegionIndex + 1) == regions.size();
        Pt.setRegionRoot(region, RootRegionIteration);
        assert(!EntryCandidates.empty() || RootRegionIteration);
        assert(!RootRegionIteration || EntryCandidates.empty());
        if (RootRegionIteration) {
          assert(EntryCandidates.empty());
          EntryCandidates.insert({RPOT.front(), 0});
        }

        llvm::dbgs() << "\nEntry candidates:\n";
        printMap(EntryCandidates);

        mlir::Block *Entry = electEntry<mlir::Block *>(
            EntryCandidates, ShortestPathFromEntry, RPOT);
        Pt.setRegionEntry(region, Entry);
        llvm::dbgs() << "\nElected entry:\n";
        Entry->printAsOperand(llvm::dbgs());
        llvm::dbgs() << "\n";

        // Now that elected the entry node we can prooced with the inlining.
        // Extract for each non-elected entry, the inlining path.
        llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>, 4>
            LateEntryPairs = getOutlinedEntries<mlir::Block *>(EntryCandidates,
                                                               region, Entry);

        // Print all the outside predecessor.
        llvm::dbgs() << "\nNon regular entry candidates found:\n";
        printPairVector(LateEntryPairs);

        // Outline the first iteration of the cycles.
        BlockSet OutlinedNodes;
        OutlinedCycle = outlineFirstIteration(LateEntryPairs, region,
                                              OutlinedNodes, Entry, Rewriter);

        // Update the parent regions to reflect the newly added nodes.
        updateParent(Pt, region, OutlinedNodes);

        // Increment region index for next iteration.
        RegionIndex++;
      }
    }
  }

  void populateCliftLoopBody(clift::LoopOp CliftLoop, BlockSet &Region,
                             mlir::Block *Entry,
                             mlir::PatternRewriter &Rewriter) const {
    // We create a clone of the blocks in the new `CliftLoop` region.
    assert(CliftLoop->getNumRegions() == 1);
    mlir::Region &LoopRegion = CliftLoop->getRegion(0);

    // Create a new empty block in the that we will use only as a
    // placeholder for inserting other blocks, then we will remove it.
    mlir::Block *EmptyBlock = Rewriter.createBlock(&LoopRegion);
    mlir::Block *PlaceholderBlock = EmptyBlock;

    // Explicitly handle the entry block, which must come first in the
    // region, and reverse insert all the other blocks since we are adding
    // before.
    for (mlir::Block *B : Region) {
      if (B != Entry) {
        B->moveBefore(PlaceholderBlock);
        PlaceholderBlock = B;
      }
    }
    Entry->moveBefore(PlaceholderBlock);

    EmptyBlock->moveBefore(Entry);
    Rewriter.setInsertionPointToEnd(EmptyBlock);
    auto loc = UnknownLoc::get(getContext());
    Rewriter.create<LLVM::BrOp>(loc, Entry);
  }

  void generateCliftGotoSuccessors(BlockSet &Region,
                                   BlockSet &CliftLoopSuccessors,
                                   mlir::Region &Reg,
                                   mlir::PatternRewriter &Rewriter,
                                   clift::LoopOp CliftLoop) const {

    // Handle the outgoing edges from the region.
    llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>, 4>
        ExitSuccessorsPairs = getExitNodePairs<mlir::Block *>(Region);

    for (const auto &[Exit, Successor] : ExitSuccessorsPairs) {

      // Create the label in the first first basic block of the root region
      // of the function.
      Rewriter.setInsertionPoint(&Reg.front(), Reg.front().begin());
      auto loc = UnknownLoc::get(getContext());
      Rewriter.setInsertionPoint(&*(Reg.op_begin()));

      clift::MakeLabelOp MakeLabel = Rewriter.create<clift::MakeLabelOp>(loc);

      // Create the label in the successor `Block`.
      Rewriter.setInsertionPointToStart(Successor);
      clift::AssignLabelOp Label =
          Rewriter.create<clift::AssignLabelOp>(loc, MakeLabel);

      // We need create a new basic block which will contain the `goto`
      // statement, and then subsistute the branch to that block.
      mlir::Block *GotoBlock = Rewriter.createBlock(&CliftLoop->getRegion(0));

      // Create the `goto` in the new trampoline block.
      Rewriter.setInsertionPointToStart(GotoBlock);
      Rewriter.create<clift::GoToOp>(loc, MakeLabel);

      // Subsitute the outgoing edges with a branch to the `goto`s
      // containing block.
      IRMapping GotoMapping;
      GotoMapping.map(Successor, GotoBlock);
      updateTerminatorOperands(Exit, GotoMapping);

      // Accumulate all the successors reached through the `clift.goto`
      // statements, so that we can later restore the edge on the control
      // flow graph in the parent region.
      CliftLoopSuccessors.insert(Successor);
    }
  }

  void generateCliftLoopSuccessors(mlir::Block *LoopParentBlock,
                                   BlockSet &CliftLoopSuccessors,
                                   mlir::PatternRewriter &Rewriter) const {
    // In the `clift.loop` parent block we insert a switch statement
    // preserving the control flow between `clift.goto` label destinations.
    Rewriter.setInsertionPointToEnd(LoopParentBlock);

    auto Loc = UnknownLoc::get(getContext());

    if (CliftLoopSuccessors.size() == 0) {
    } else if (CliftLoopSuccessors.size() == 1) {
      mlir::Block *FirstSuccessor = *CliftLoopSuccessors.begin();
      Rewriter.create<LLVM::BrOp>(Loc, FirstSuccessor);
    } else if (CliftLoopSuccessors.size() == 2) {
      mlir::Value ConstantValue =
          Rewriter.create<LLVM::ConstantOp>(Loc, Rewriter.getBoolAttr(false));
      mlir::Block *FirstSuccessor = *CliftLoopSuccessors.begin();
      mlir::Block *SecondSuccessor = *std::next(CliftLoopSuccessors.begin());
      Rewriter.create<LLVM::CondBrOp>(Loc, ConstantValue, FirstSuccessor,
                                      SecondSuccessor);
    } else {
      // TODO: implement this with a `LLVM::SwitchOp` operation.
      assert(false);
    }
  }

  void generateCliftContinue(BlockSet &Region, mlir::Block *Entry,
                             mlir::PatternRewriter &Rewriter,
                             clift::LoopOp CliftLoop) const {

    // Insert the `clift.continue` operation.
    llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>, 4>
        ContinueNodePairs = getContinueNodePairs<mlir::Block *>(Entry, Region);
    for (const auto &[Continue, Entry] : ContinueNodePairs) {

      // Creation of a block that will contain the `clift.continue`
      // operation.
      mlir::Block *ContinueBlock =
          Rewriter.createBlock(&CliftLoop->getRegion(0));

      // Create the `clift.continue` operation.
      Rewriter.setInsertionPointToStart(ContinueBlock);
      auto loc = UnknownLoc::get(getContext());
      Rewriter.create<clift::ContinueOp>(loc);

      // Substitute the retreating edges to the entry with a branch to the
      // `continue` containing block.
      IRMapping ContinueMapping;
      ContinueMapping.map(Entry, ContinueBlock);
      updateTerminatorOperands(Continue, ContinueMapping);
    }
  }

  void updateParentWithCliftLoop(BlockSet &Region,
                                 revng::detail::ParentTree<mlir::Block *> &Pt,
                                 mlir::Block *LoopParentBlock) const {
    // Update in the parent region the status of the nodes.
    if (Pt.hasParent(Region)) {

      // Insert in the parent region the block containing the `clift.loop`.
      BlockSet &ParentRegion = Pt.getParent(Region);
      ParentRegion.insert(LoopParentBlock);

      // Remove from the parent region all the blocks that now constitute
      // the body of the `clift.loop`.
      for (mlir::Block *B : Region) {
        ParentRegion.erase(B);
      }
    }
  }

  void performCliftLoopGeneration(
      mlir::Region &Reg, mlir::PatternRewriter &Rewriter,
      revng::detail::ParentTree<mlir::Block *> &Pt) const {

    // Perform `clift.loop` generation.
    size_t RegionIndex = 0;
    for (BlockSet region : Pt.regions()) {
      if (Pt.isRegionRoot(region) == true) {
        continue;
      }

      // Obtain the parent region of the function we are restructuring.
      mlir::Region *ParentRegion = nullptr;
      for (mlir::Block *B : region) {
        if (ParentRegion == nullptr) {
          ParentRegion = B->getParent();
        }
        assert(B->getParent() == ParentRegion);
      }

      // Create a new block to contain the `clift.loop` operation.
      mlir::Block *LoopParentBlock = Rewriter.createBlock(ParentRegion);

      // Retrieve the elected entry block.
      mlir::Block *Entry = Pt.getRegionEntry(region);

      // Connect the block containing the `clift.loop` to the old
      // predecessors.
      IRMapping EntryMapping;
      EntryMapping.map(Entry, LoopParentBlock);
      llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>, 4>
          PredecessorNodePairs =
              getLoopPredecessorNodePairs<mlir::Block *>(Entry, region);
      for (const auto &[Predecessor, EntryCandidate] : PredecessorNodePairs) {
        assert(EntryCandidate == Entry);
        updateTerminatorOperands(Predecessor, EntryMapping);
      }

      // Create a new `clift.loop` operation.
      Rewriter.setInsertionPointToStart(LoopParentBlock);
      auto loc = UnknownLoc::get(getContext());
      clift::LoopOp CliftLoop = Rewriter.create<clift::LoopOp>(loc);

      populateCliftLoopBody(CliftLoop, region, Entry, Rewriter);

      BlockSet CliftLoopSuccessors;
      generateCliftGotoSuccessors(region, CliftLoopSuccessors, Reg, Rewriter,
                                  CliftLoop);

      generateCliftLoopSuccessors(LoopParentBlock, CliftLoopSuccessors,
                                  Rewriter);

      generateCliftContinue(region, Entry, Rewriter, CliftLoop);

      updateParentWithCliftLoop(region, Pt, LoopParentBlock);

      // Increment region index for next iteration.
      RegionIndex++;
    }
  }

  void performRestructureCliftRegion(mlir::Region &reg,
                                     mlir::PatternRewriter &rewriter) const {

    // Declare the global `ParentTree` object which will contain the region
    // identified in the current function.
    revng::detail::ParentTree<mlir::Block *> Pt;

    // Perform region identification.
    performRegionIdentification(reg, rewriter, Pt);

    // Perform `clift.loop` generation.
    performCliftLoopGeneration(reg, rewriter, Pt);
  }
};

struct RestructureClift : public impl::RestructureCliftBase<RestructureClift> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RestructureCliftRewriter>(&getContext());

    SmallVector<Operation *> Functions;
    getOperation()->walk([&](LLVM::LLVMFuncOp F) { Functions.push_back(F); });

    auto Strictness = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(
            applyOpPatternsAndFold(Functions, std::move(patterns), Strictness)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createRestructureCliftPass() {
  return std::make_unique<RestructureClift>();
}
