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
#include "llvm/ADT/STLExtras.h"
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

  using ParentTree = revng::detail::ParentTree<mlir::Block *>;

  using RegionNode = revng::detail::RegionNode<mlir::Block *>;
  using RegionNodePointerPair =
      revng::detail::RegionNodePointerPair<mlir::Block *>;
  using RegionTree = revng::detail::RegionTree<mlir::Block *>;

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

  void printRegionNode(RegionNode &RegionNode) const {
    for (RegionNode::NodeRef &Block : RegionNode) {
      if (std::holds_alternative<mlir::Block *>(Block)) {
        mlir::Block *B = std::get<mlir::Block *>(Block);
        B->printAsOperand(llvm::dbgs());
        llvm::dbgs() << "\n";
      } else if (std::holds_alternative<RegionNodePointerPair>(Block)) {
        RegionNodePointerPair Pair = std::get<RegionNodePointerPair>(Block);
        llvm::dbgs() << "Subregion ID: " << Pair.first << "\n";
      }
    }
  }

  void printRegionTree(RegionTree &RegionTree) const {
    llvm::dbgs() << "\nRegionTree:\n";
    size_t RegionIndex = 0;
    for (RegionNode &RegionNode : RegionTree.regions()) {
      llvm::dbgs() << "Region idx: " << RegionIndex << " composed by nodes:\n";
      printRegionNode(RegionNode);
    }
    RegionIndex++;
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

  void updateParent(ParentTree &Pt, BlockSet &Region,
                    BlockSet &OutlinedNodes) const {

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

  void performRegionIdentification(mlir::Region &FunctionRegion,
                                   mlir::PatternRewriter &Rewriter,
                                   RegionTree &Rt) const {

    // TODO: Remove this `ParentTree`. It is now kept to simplify the transition
    // from `ParentTree` to `RegionTree` usage.
    ParentTree Pt;

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

      // New Impl. starting here.
      // Compute the Reverse Post Order.
      llvm::SmallVector<mlir::Block *> RPOT;
      using RPOTraversal = llvm::ReversePostOrderTraversal<mlir::Region *>;
      llvm::copy(RPOTraversal(&FunctionRegion), std::back_inserter(RPOT));

      // Compute the distance of each node from the entry node.
      llvm::DenseMap<mlir::Block *, size_t> ShortestPathFromEntry =
          computeDistanceFromEntry(&FunctionRegion);

      // The following routine pre-computes the entry block of each region.
      size_t RegionIndex = 0;
      for (BlockSet &Region : Pt.regions()) {
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

        mlir::Block *Entry = electEntry<mlir::Block *>(
            EntryCandidates, ShortestPathFromEntry, RPOT);
        Pt.setRegionEntry(Region, Entry);
        RegionIndex++;
      }

      // Transpile the already ordered regions into a `RegionTree`.
      RegionIndex = 0;
      std::map<BlockSet *, size_t> RegionIDMap;
      for (BlockSet &Region : Pt.regions()) {

        // Populate the map containing the correspondance between region and
        // index.
        RegionIDMap[&Region] = RegionIndex;

        // Create the `RegionNode` object.
        RegionNode RegionNode(Rt);

        // Insert in the `RegionNode` the entry at first, and then the other
        // nodes.
        mlir::Block *Entry = Pt.getRegionEntry(Region);
        RegionNode.insertElementEntry(Entry);
        for (mlir::Block *B : Region) {
          if (B != Entry) {
            RegionNode.insertElement(B);
          }
        }

        // Insert the `RegionNode` in the `RegionTree` object.
        Rt.insertRegion(std::move(RegionNode));
        RegionIndex++;
      }

      // Cycling through the original regions, and exploiting the `ParentMap`
      // relationship we substitute in the regions the nodes by the index
      // representing the nested region.
      for (BlockSet &Region : Pt.regions()) {

        // Check that the region we are analyzing actually has a `Parent`.
        if (Pt.hasParent(Region)) {

          // We obtain the ID of the `Parent` region, exploting the map data
          // structure that we kept for this purpose.
          size_t ParentIndex = RegionIDMap.at(&Pt.getParent(Region));

          // We obtain the `Parent` region from the `RegionTree` data structure.
          RegionNode &ParentRegion = Rt.getRegion(ParentIndex);

          // Remove from the `Parent` region all the blocks that belong to the
          // current region. If we happen to remove the entry block from a
          // region, the subsequent insertion of the `RegionIndex` will become
          // the entry of the region.
          bool IsEntry = false;
          for (mlir::Block *Block : Region) {
            IsEntry = ParentRegion.eraseElement(Block);
          }

          // Insert in the `Parent` region the ID representing the current child
          // region.
          size_t CurrentIndex = RegionIDMap.at(&Region);
          RegionNodePointerPair RegionNodePointerPair =
              std::make_pair(CurrentIndex, &Rt);
          if (IsEntry) {
            ParentRegion.insertElementEntry(RegionNodePointerPair);
          } else {
            ParentRegion.insertElement(RegionNodePointerPair);
          }
        }
      }

      // Output as debug the `RegionTree` structure.
      printRegionTree(Rt);

      // Instantiate a postorder visit on the `RegionTree` in order to perform
      // the first iteration outlining.
      // We start the visit from the last node in the `RegionTree`, which is
      // always the `root` region.
      for (RegionNode *Region : post_order(&*(Rt.rbegin()))) {

        // We now perform the first iteration outlining procedure. The
        // outlining is morally performed by the parent region for its
        // children regions.
        for (RegionNode *ChildRegion : Region->successor_range()) {

          BlockSet NodesSet = ChildRegion->getBlocksSet();
          mlir::Block *Entry = ChildRegion->getEntryBlock();

          llvm::DenseMap<mlir::Block *, size_t> EntryCandidates =
              getEntryCandidates<mlir::Block *>(NodesSet);

          llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>>
              LateEntryPairs = getOutlinedEntries<mlir::Block *>(
                  EntryCandidates, NodesSet, Entry);

          // Print all the outside predecessor.
          llvm::dbgs() << "\nNon regular entry candidates found:\n";
          printPairVector(LateEntryPairs);

          // Outline the first iteration of the cycles.
          BlockSet OutlinedNodes;
          OutlinedCycle |= outlineFirstIteration(
              LateEntryPairs, NodesSet, OutlinedNodes, Entry, Rewriter);

          // The outlined nodes must be added to the parent region with respect
          // to the one they were extracted form, that is the region we are
          // iterating onto.
          for (mlir::Block *OutlinedBlock : OutlinedNodes) {
            Region->insertElement(OutlinedBlock);
          }
        }
      }
    }
  }

  clift::LoopOp
  generateCliftLoop(BlockSet &Region, mlir::Block *Entry,
                    llvm::SmallVector<mlir::Block *> &CliftLoopSuccessors,
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
    clift::LoopOp CliftLoop =
        Rewriter.create<clift::LoopOp>(Loc, CliftLoopSuccessors);
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
    for (mlir::Block &CliftLoopBlock : LoopRegion) {
      for (mlir::Block *Predecessor : predecessor_range(&CliftLoopBlock)) {
        assert(Predecessor->getParent() == &LoopRegion);
      }
    }
  }

  void generateCliftGotoSuccessors(
      BlockSet &Region, mlir::Region &FunctionRegion,
      llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>>
          &ExitSuccessorsPairs,
      mlir::PatternRewriter &Rewriter, clift::LoopOp CliftLoop) const {
    // Handle the outgoing edges from the region.
    mlir::Region &LoopRegion = CliftLoop->getRegion(0);

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
      mlir::Block *GotoBlock = Rewriter.createBlock(&LoopRegion);

      // Create the `goto` in the new trampoline block.
      Rewriter.setInsertionPointToStart(GotoBlock);
      Rewriter.create<clift::GoToOp>(Loc, MakeLabel);

      // Subsitute the outgoing edges with a branch to the `goto`s
      // containing block.
      IRMapping GotoMapping;
      GotoMapping.map(Successor, GotoBlock);
      updateTerminatorOperands(Exit, GotoMapping);
    }

    // Additional check that all the successors of each block now living inside
    // the `clift.loop`, is also in the same region.
    for (mlir::Block &CliftLoopBlock : LoopRegion) {
      for (mlir::Block *Successor : successor_range(&CliftLoopBlock)) {
        assert(Successor->getParent() == &LoopRegion);
      }
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

  void updateParentWithCliftLoop(RegionNode *ParentRegion,
                                 RegionNode *ChildRegion, RegionTree &Rt,
                                 clift::LoopOp CliftLoop) const {

    // We need to take care of the fact that, if the `ChildRegion` under
    // analysis is the entry of the parent region, we need to insert the new
    // block containing the `clift.loop` as entry.
    mlir::Block *CliftLoopBlock = CliftLoop->getBlock();

    if (ParentRegion->isChildRegionEntry(ChildRegion)) {
      ParentRegion->insertElementEntry(CliftLoopBlock);
    } else {
      ParentRegion->insertElement(CliftLoopBlock);
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

  void performCliftLoopGeneration(mlir::Region &FunctionRegion,
                                  mlir::PatternRewriter &Rewriter,
                                  RegionTree &Rt) const {

    // Perform the `clift.loop` generation, by instantiating a `post_order`
    // visit over the `ParentTree`, and by generating the `clift.loop` for all
    // the children of the `RegionNode` under visit.
    for (RegionNode *Region : post_order(&*(Rt.rbegin()))) {
      for (RegionNode *ChildRegion : Region->successor_range()) {
        BlockSet NodesSet = ChildRegion->getBlocksSet();
        mlir::Block *Entry = ChildRegion->getEntryBlock();

        // Precompute the edges exiting from the region.
        llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>>
            ExitSuccessorsPairs = getExitNodePairs<mlir::Block *>(NodesSet);
        llvm::SmallVector<mlir::Block *> CliftLoopSuccessors(
            llvm::make_second_range(ExitSuccessorsPairs));

        // Perform the `clift.loop` transformations.
        clift::LoopOp CliftLoop =
            generateCliftLoop(NodesSet, Entry, CliftLoopSuccessors, Rewriter);

        populateCliftLoopBody(CliftLoop, NodesSet, Entry, Rewriter);

        generateCliftGotoSuccessors(NodesSet, FunctionRegion,
                                    ExitSuccessorsPairs, Rewriter, CliftLoop);

        generateCliftContinue(NodesSet, Entry, Rewriter, CliftLoop);

        updateParentWithCliftLoop(Region, ChildRegion, Rt, CliftLoop);

        generateCliftRetreating(Entry, Rewriter, CliftLoop);
      }
    }
  }

  void performRestructureCliftRegion(mlir::Region &FunctionRegion,
                                     mlir::PatternRewriter &Rewriter) const {

    // Declare the global `RegionTree` object which will contain the regions
    // identified in the current function.
    RegionTree Rt;

    // Perform region identification.
    performRegionIdentification(FunctionRegion, Rewriter, Rt);

    // Perform `clift.loop` generation.
    performCliftLoopGeneration(FunctionRegion, Rewriter, Rt);
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
