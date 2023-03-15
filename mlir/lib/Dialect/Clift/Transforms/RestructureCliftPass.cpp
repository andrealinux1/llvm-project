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
  using BlockSetVect = llvm::SmallVector<BlockSet>;
  using BlockIntMap = llvm::DenseMap<mlir::Block *, size_t>;
  using BlockVect = llvm::SmallVector<mlir::Block *>;

  using mlir::OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp Op,
                  mlir::PatternRewriter &Rewriter) const final {
    // llvm::dbgs() << "Invoking matchAndRewrite on " << "\n";
    // op->dump();

    // Ensure that we start from a `LLVMFuncOp` with a single `cf` region.
    assert(Op->getNumRegions() == 1);

    // Transform only regions which have an actual size.
    mlir::Region &FunctionRegion = Op->getRegion(0);
    if (not FunctionRegion.getBlocks().empty()) {
      performRestructureCliftRegion(FunctionRegion, Rewriter);
    }
    return success();
  }

  void printBackedge(EdgeDescriptor &Backedge) const {
    llvm::dbgs() << "Backedge: ";
    Backedge.first->printAsOperand(llvm::dbgs());
    llvm::dbgs() << " -> ";
    Backedge.second->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  void printBackedges(EdgeSet &Backedges) const {
    for (EdgeDescriptor Backedge : Backedges) {
      printBackedge(Backedge);
    }
  }

  void printReachableBlocks(EdgeSet &Backedges) const {
    for (EdgeDescriptor Backedge : Backedges) {
      printBackedge(Backedge);
      llvm::dbgs() << "We can reach blocks:\n";
      for (mlir::Block *Reachable :
           nodesBetween(Backedge.second, Backedge.first)) {
        Reachable->printAsOperand(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
    }
  }

  void printRegions(BlockSetVect &Regions) const {
    size_t Regionindex = 0;
    for (BlockSet &Region : Regions) {
      llvm::dbgs() << "Region idx: " << Regionindex << " composed by nodes: \n";
      for (mlir::Block *Block : Region) {
        Block->printAsOperand(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
      Regionindex++;
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

  // Returns true if during this iteration we outline a loop construct.
  bool outlineFirstIteration(
      llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>>
          &LateEntryPairs,
      BlockSet &Region, BlockSet &OutlinedNodes, mlir::Block *Entry,
      mlir::PatternRewriter &Rewriter) const {
    bool OutlinedCycle = false;

    for (auto const &[Predecessor, LateEntry] : LateEntryPairs) {
      using GT = typename llvm::GraphTraits<mlir::Block *>;
      using StateType = typename revng::detail::DFSBackedgeState<mlir::Block *>;

      //  Enqueue the successors. Note that if the successor is the elected
      // entry, do not clone it, because it is correct to jump there. If the
      // successor is outside of the current set region, do not clone it
      // either, this path will be represented with `goto`s at the current
      // stage.
      StateType State([&Region, Entry](const mlir::Block *Block) {
        return Region.contains(Block) && Block != Entry;
      });

      BlockSet IterationOutlinedNodes;
      IRMapping CloneMapping;
      IRMapping LateEntryOnlyMapping;

      using odf_iterator = llvm::df_iterator<mlir::Block *, StateType, true>;
      auto Begin = odf_iterator::begin(LateEntry, State);
      auto End = odf_iterator::end(LateEntry, State);

      for (mlir::Block *Block : llvm::make_range(Begin, End)) {

        // Manual cloning of the block body.
        IRMapping Mapping;
        mlir::Block *BlockClone = Rewriter.createBlock(Block);
        Mapping.map(Block, BlockClone);
        CloneMapping.map(Block, BlockClone);

        // Add the outlined nodes to the set that we will return.
        OutlinedNodes.insert(BlockClone);

        // Add the outlined nodes to a set to update the terminators.
        IterationOutlinedNodes.insert(BlockClone);

        for (auto &BlockOp : *Block) {
          Operation *Clone = Rewriter.clone(BlockOp, Mapping);
          Mapping.map(BlockOp.getResults(), Clone->getResults());
        }

        // Remap block successors that have been already cloned on the
        // respective clone (otherwise we could end up in outlining loops).
        updateTerminatorOperands(BlockClone, CloneMapping);

        // Detect a loop, and in case
        for (mlir::Block *Successor :
             llvm::make_range(GT::child_begin(Block), GT::child_end(Block))) {
          if (State.onStack(Successor)) {
            OutlinedCycle |= true;
          }
        }
      }

      // Remap the terminator successors in order to point to the cloned nodes.
      for (mlir::Block *BlockClone : IterationOutlinedNodes) {
        updateTerminatorOperands(BlockClone, CloneMapping);
      }

      // Remap the terminator of the predecessor node to point to the outlined
      // iteration. We need to manually extract from the `CloneMapping` only the
      // entry relative to the `LateEntry` block, or we could wrongly map other
      // branches to their clones extracted during this iteration.
      LateEntryOnlyMapping.map(LateEntry, CloneMapping.lookup(LateEntry));
      updateTerminatorOperands(Predecessor, LateEntryOnlyMapping);
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
      mlir::Region &FunctionRegion, mlir::PatternRewriter &Rewriter,
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

      EdgeSet Backedges = getBackedges(&FunctionRegion.front());
      llvm::dbgs() << "\nInitial backedges:\n";
      printBackedges(Backedges);
      llvm::dbgs() << "\nInitial reachables:\n";
      printReachableBlocks(Backedges);

      // Declare a metaobject which will contain all the identified region
      // objects.
      BlockSetVect Regions;

      // Push manually the `root` region in the identified regions.
      BlockSet RootRegion;
      for (mlir::Block &B : FunctionRegion) {
        RootRegion.insert(&B);
      }
      Regions.push_back(RootRegion);

      // Collect all the identified regions in a single vector.
      for (EdgeDescriptor Backedge : Backedges) {
        BlockSet RegionNodes = nodesBetween(Backedge.second, Backedge.first);
        Regions.push_back(std::move(RegionNodes));
      }

      llvm::dbgs() << "\nInitial regions:\n";
      printRegions(Regions);

      // Simplify the identified regions in order to satisfy the requirements.
      simplifyRegions(Regions);
      llvm::dbgs() << "\nAfter simplification:\n";
      printRegions(Regions);

      // Order the identified regions.
      sortRegions(Regions);
      llvm::dbgs() << "\nAfter sorting:\n";
      printRegions(Regions);

      // Insert the regions in the region tree.
      for (BlockSet &Region : Regions) {
        Pt.insertRegion(Region);
      }

      // Order the regions inside the `ParentTree`. This invokes region
      // reordering
      Pt.order();

      // Order the regions so that they go from the outer one to the inner one.
      llvm::dbgs() << "\nAfter ordering:\n";
      printRegions(Pt.getRegions());

      // Compute the Reverse Post Order.
      llvm::SmallVector<mlir::Block *> RPOT;
      using RPOTraversal = llvm::ReversePostOrderTraversal<mlir::Region *>;
      llvm::copy(RPOTraversal(&FunctionRegion), std::back_inserter(RPOT));

      // Compute the distance of each node from the entry node.
      llvm::DenseMap<mlir::Block *, size_t> ShortestPathFromEntry =
          computeDistanceFromEntry(&FunctionRegion);

      size_t RegionIndex = 0;
      for (BlockSet &Region : Pt.regions()) {
        llvm::dbgs() << "\nRestructuring region idx: " << RegionIndex << ":\n";
        llvm::DenseMap<mlir::Block *, size_t> EntryCandidates =
            getEntryCandidates<mlir::Block *>(Region);

        // In case we are analyzing the root region, we expect to have no entry
        // candidates.
        bool RootRegionIteration = (RegionIndex + 1) == Regions.size();
        Pt.setRegionRoot(Region, RootRegionIteration);
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
        Pt.setRegionEntry(Region, Entry);
        llvm::dbgs() << "\nElected entry:\n";
        Entry->printAsOperand(llvm::dbgs());
        llvm::dbgs() << "\n";

        // Now that elected the entry node we can prooced with the inlining.
        // Extract for each non-elected entry, the inlining path.
        llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>>
            LateEntryPairs = getOutlinedEntries<mlir::Block *>(EntryCandidates,
                                                               Region, Entry);

        // Print all the outside predecessor.
        llvm::dbgs() << "\nNon regular entry candidates found:\n";
        printPairVector(LateEntryPairs);

        // Outline the first iteration of the cycles.
        BlockSet OutlinedNodes;
        OutlinedCycle = outlineFirstIteration(LateEntryPairs, Region,
                                              OutlinedNodes, Entry, Rewriter);

        // Update the parent regions to reflect the newly added nodes.
        updateParent(Pt, Region, OutlinedNodes);

        // Increment region index for next iteration.
        RegionIndex++;
      }
    }
  }

  clift::LoopOp generateCliftLoop(BlockSet &Region, mlir::Block *Entry,
                                  mlir::PatternRewriter &Rewriter) const {
    // Obtain the parent region of the function we are restructuring.
    mlir::Region *ParentRegion = nullptr;
    for (mlir::Block *B : Region) {
      if (ParentRegion == nullptr) {
        ParentRegion = B->getParent();
      }
      assert(B->getParent() == ParentRegion);
    }

    // Create a new block to contain the `clift.loop` operation.
    mlir::Block *LoopParentBlock = Rewriter.createBlock(ParentRegion);

    // Connect the block containing the `clift.loop` to the old
    // predecessors.
    IRMapping EntryMapping;
    EntryMapping.map(Entry, LoopParentBlock);
    llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>>
        PredecessorNodePairs =
            getLoopPredecessorNodePairs<mlir::Block *>(Entry, Region);
    for (const auto &[Predecessor, EntryCandidate] : PredecessorNodePairs) {
      assert(EntryCandidate == Entry);
      updateTerminatorOperands(Predecessor, EntryMapping);
    }

    // Create a new `clift.loop` operation.
    Rewriter.setInsertionPointToStart(LoopParentBlock);
    auto Loc = UnknownLoc::get(getContext());
    clift::LoopOp CliftLoop = Rewriter.create<clift::LoopOp>(Loc);
    return CliftLoop;
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
    auto Loc = UnknownLoc::get(getContext());
    Rewriter.create<LLVM::BrOp>(Loc, Entry);

    // Additional check that the predecessor of each block now living inside
    // the `clift.loop`, is also in the same region.
    using GT = llvm::GraphTraits<llvm::Inverse<mlir::Block *>>;
    for (mlir::Block &CliftLoopBlock : LoopRegion) {
      for (mlir::Block *Predecessor :
           llvm::make_range(GT::child_begin(&CliftLoopBlock),
                            GT::child_end(&CliftLoopBlock))) {
        assert(Predecessor->getParent() == &LoopRegion);
      }
    }
  }

  void generateCliftGotoSuccessors(BlockSet &Region,
                                   BlockSet &CliftLoopSuccessors,
                                   mlir::Region &FunctionRegion,
                                   mlir::PatternRewriter &Rewriter,
                                   clift::LoopOp CliftLoop) const {
    // Handle the outgoing edges from the region.
    llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>>
        ExitSuccessorsPairs = getExitNodePairs<mlir::Block *>(Region);

    for (const auto &[Exit, Successor] : ExitSuccessorsPairs) {

      // Create the label in the first basic block of the root region of the
      // function.
      Rewriter.setInsertionPoint(&FunctionRegion.front(),
                                 FunctionRegion.front().begin());
      auto Loc = UnknownLoc::get(getContext());

      clift::MakeLabelOp MakeLabel = Rewriter.create<clift::MakeLabelOp>(Loc);

      // Assign the label in the successor `Block`.
      Rewriter.setInsertionPointToStart(Successor);
      Rewriter.create<clift::AssignLabelOp>(Loc, MakeLabel);

      // We need create a new basic block which will contain the `goto`
      // statement, and then subsistute the branch to that block.
      mlir::Block *GotoBlock = Rewriter.createBlock(&CliftLoop->getRegion(0));

      // Create the `goto` in the new trampoline block.
      Rewriter.setInsertionPointToStart(GotoBlock);
      Rewriter.create<clift::GoToOp>(Loc, MakeLabel);

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

    // Additional check that all the successors of each block now living inside
    // the `clift.loop`, is also in the same region.
    mlir::Region &LoopRegion = CliftLoop->getRegion(0);
    using GT = llvm::GraphTraits<mlir::Block *>;
    for (mlir::Block &CliftLoopBlock : LoopRegion) {
      for (mlir::Block *Successor :
           llvm::make_range(GT::child_begin(&CliftLoopBlock),
                            GT::child_end(&CliftLoopBlock))) {
        assert(Successor->getParent() == &LoopRegion);
      }
    }
  }

  void generateCliftLoopSuccessors(clift::LoopOp CliftLoop,
                                   BlockSet &CliftLoopSuccessors,
                                   mlir::PatternRewriter &Rewriter) const {
    // In the `clift.loop` parent block we insert a switch statement
    // preserving the control flow between `clift.goto` label destinations.
    Rewriter.setInsertionPointAfter(CliftLoop);

    auto Loc = UnknownLoc::get(getContext());

    if (CliftLoopSuccessors.size() == 0) {

      // Even if we don't have any edge exiting from the `clift.loop`, we need
      // to manually place a `UnreachableInst` after the `clift.loop`.
      Rewriter.create<LLVM::UnreachableOp>(Loc);
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
    llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>>
        ContinueNodePairs = getContinueNodePairs<mlir::Block *>(Entry, Region);
    for (const auto &[Continue, Entry] : ContinueNodePairs) {

      // Creation of a block that will contain the `clift.continue`
      // operation.
      mlir::Block *ContinueBlock =
          Rewriter.createBlock(&CliftLoop->getRegion(0));

      // Create the `clift.continue` operation.
      Rewriter.setInsertionPointToStart(ContinueBlock);
      auto Loc = UnknownLoc::get(getContext());
      Rewriter.create<clift::ContinueOp>(Loc);

      // Substitute the retreating edges to the entry with a branch to the
      // `continue` containing block.
      IRMapping ContinueMapping;
      ContinueMapping.map(Entry, ContinueBlock);
      updateTerminatorOperands(Continue, ContinueMapping);
    }
  }

  void updateParentWithCliftLoop(BlockSet &Region,
                                 revng::detail::ParentTree<mlir::Block *> &Pt,
                                 clift::LoopOp CliftLoop) const {

    // We recursively update the parent regions until we reach the roo one.
    // TODO: this is a temporary hack, `ParentTree` new design should handle
    //       this for us.
    BlockSet *ParentRegion = &Pt.getParent(Region);
    while (ParentRegion != nullptr) {

      // Insert in the parent region the block containing the `clift.loop`.
      mlir::Block *LoopParentBlock = CliftLoop->getBlock();
      ParentRegion->insert(LoopParentBlock);

      // Remove from the parent region all the blocks that now constitute
      // the body of the `clift.loop`.
      for (mlir::Block *B : Region) {
        ParentRegion->erase(B);
      }

      // Retrieve the parent region, if any.
      if (Pt.hasParent(*ParentRegion)) {
        ParentRegion = &Pt.getParent(*ParentRegion);
      } else {
        ParentRegion = nullptr;
      }
    }
  }

  void generateCliftRetreating(mlir::Block *Entry,
                               mlir::PatternRewriter &Rewriter,
                               clift::LoopOp CliftLoop) const {
    // Generate the abnormal retreating flows with the use of a couple of a
    // `clift.label` and `clift.goto`.
    EdgeSet LoopBackedges = getBackedges(Entry);
    llvm::dbgs() << "\nAbnormal retreating backedges:\n";
    printBackedges(LoopBackedges);

    for (EdgeDescriptor RetreatingEdge : LoopBackedges) {
      mlir::Block *Source = RetreatingEdge.first;
      mlir::Block *Target = RetreatingEdge.second;

      // We should not find an abnormal retreating pointing to the entry of
      // the loop, we should have already transformed that into a
      // `clift.continue`.
      assert(Target != Entry);

      // Create the label in the first basic block of the `clift.loop`.
      mlir::Region &LoopRegion = CliftLoop->getRegion(0);
      Rewriter.setInsertionPoint(&LoopRegion.front(),
                                 LoopRegion.front().begin());
      auto Loc = UnknownLoc::get(getContext());

      clift::MakeLabelOp MakeLabel = Rewriter.create<clift::MakeLabelOp>(Loc);

      // Assign the label in the `Target` block.
      Rewriter.setInsertionPointToStart(Target);
      Rewriter.create<clift::AssignLabelOp>(Loc, MakeLabel);

      // We need to create a new basic block containing the `clift.goto`, and
      // then substitute the branch to this block.
      mlir::Block *GotoBlock = Rewriter.createBlock(&LoopRegion);

      // Create the `clift.goto` in the trampoline block.
      Rewriter.setInsertionPointToStart(GotoBlock);
      Rewriter.create<clift::GoToOp>(Loc, MakeLabel);

      // Remap the branches to the `clift.goto` block.
      IRMapping GotoMapping;
      GotoMapping.map(Target, GotoBlock);
      updateTerminatorOperands(Source, GotoMapping);
    }
  }

  void performCliftLoopGeneration(
      mlir::Region &FunctionRegion, mlir::PatternRewriter &Rewriter,
      revng::detail::ParentTree<mlir::Block *> &Pt) const {

    // Perform `clift.loop` generation.
    size_t RegionIndex = 0;
    for (BlockSet Region : Pt.regions()) {
      if (Pt.isRegionRoot(Region) == true) {
        continue;
      }

      // Retrieve the elected entry block.
      mlir::Block *Entry = Pt.getRegionEntry(Region);

      clift::LoopOp CliftLoop = generateCliftLoop(Region, Entry, Rewriter);

      populateCliftLoopBody(CliftLoop, Region, Entry, Rewriter);

      BlockSet CliftLoopSuccessors;
      generateCliftGotoSuccessors(Region, CliftLoopSuccessors, FunctionRegion,
                                  Rewriter, CliftLoop);

      generateCliftLoopSuccessors(CliftLoop, CliftLoopSuccessors, Rewriter);

      generateCliftContinue(Region, Entry, Rewriter, CliftLoop);

      updateParentWithCliftLoop(Region, Pt, CliftLoop);

      generateCliftRetreating(Entry, Rewriter, CliftLoop);

      // Increment region index for next iteration.
      RegionIndex++;
    }
  }

  void performRestructureCliftRegion(mlir::Region &FunctionRegion,
                                     mlir::PatternRewriter &Rewriter) const {

    // Declare the global `ParentTree` object which will contain the region
    // identified in the current function.
    revng::detail::ParentTree<mlir::Block *> Pt;

    // Perform region identification.
    performRegionIdentification(FunctionRegion, Rewriter, Pt);

    // Perform `clift.loop` generation.
    performCliftLoopGeneration(FunctionRegion, Rewriter, Pt);
  }
};

struct RestructureClift : public impl::RestructureCliftBase<RestructureClift> {
  void runOnOperation() override {
    RewritePatternSet Patterns(&getContext());
    Patterns.add<RestructureCliftRewriter>(&getContext());

    SmallVector<Operation *> Functions;
    getOperation()->walk([&](LLVM::LLVMFuncOp F) { Functions.push_back(F); });

    auto Strictness = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(
            applyOpPatternsAndFold(Functions, std::move(Patterns), Strictness)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createRestructureCliftPass() {
  return std::make_unique<RestructureClift>();
}
