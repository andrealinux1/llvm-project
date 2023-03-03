#include "mlir/Dialect/Arith/IR/Arith.h"
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

// Provide graph

using namespace mlir;

namespace {

class RestructureCFRewriter : public OpRewritePattern<LLVM::LLVMFuncOp> {
  using EdgeDescriptor = revng::detail::EdgeDescriptor<mlir::Block *>;
  using EdgeSet = llvm::SmallSet<EdgeDescriptor, 4>;
  using BlockSet = llvm::SmallPtrSet<mlir::Block *, 4>;
  using BlockSetVect = llvm::SmallVector<BlockSet, 4>;
  using BlockIntMap = llvm::DenseMap<mlir::Block *, size_t>;
  using BlockVect = llvm::SmallVector<mlir::Block *, 4>;

  using mlir::OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(LLVM::LLVMFuncOp op,
                                      mlir::PatternRewriter &rewriter) const final {
    //llvm::dbgs() << "Invoking matchAndRewrite on " << "\n";
    //op->dump();

    // Ensure that we start from a `LLVMFuncOp` with a single `cf` region.
    assert(op->getNumRegions() == 1);

    // Transform only regions which have an actual size.
    mlir::Region &reg = op->getRegion(0);
    if (not reg.getBlocks().empty()) {
      //Reg.viewGraph();
      performRestructureCFRegion(reg, rewriter);
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
      for (mlir::Block *reachable : nodesBetween(backedge.second,
                                                 backedge.first)) {
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

    void printPairVector(llvm::SmallVectorImpl<std::pair<mlir::Block *, mlir::Block *>> &Vector) const {
    for (auto const &[First, Second] : Vector) {
      First->printAsOperand(llvm::dbgs());
      llvm::dbgs() << " -> ";
      Second->printAsOperand(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  }

  template<class NodeT>
  size_t mapAt(llvm::DenseMap<NodeT, size_t> &Map, NodeT Element) const {
    auto MapIt = Map.find(Element);
    assert(MapIt != Map.end());
    return MapIt->second;
  }

  template<class NodeT>
  NodeT electEntry(llvm::DenseMap<NodeT, size_t> &EntryCandidates,
                   llvm::DenseMap<NodeT, size_t> &ShortestPathFromEntry,
                   llvm::SmallVectorImpl<NodeT> &RPOT) const {
    // Elect the Entry as the the candidate entry with the largest number of
    // incoming edges from outside the region.
    // If there's a tie, i.e. there are 2 or more candidate entries with the
    // same number of incoming edges from an outer region, we select the entry
    // with the minimal shortest path from entry.
    // It it's still a tie, i.e. there are 2 or more candidate entries with the
    // same number of incoming edges from an outer region and the same minimal
    // shortest path from entry, then we disambiguate by picking the entry that
    // comes first in RPOT.
    NodeT Entry = Entry = EntryCandidates.begin()->first;
    {
      size_t MaxNEntries = EntryCandidates.begin()->second;
      auto ShortestPathIt = ShortestPathFromEntry.find(Entry);
      assert(ShortestPathIt != ShortestPathFromEntry.end());
      size_t ShortestPath = mapAt(ShortestPathFromEntry, Entry);
      auto EntriesEnd = EntryCandidates.end();
      for (NodeT Block : RPOT) {
        auto EntriesIt = EntryCandidates.find(Block);
        if (EntriesIt != EntriesEnd) {
          const auto &[EntryCandidate, NumIncoming] = *EntriesIt;
          if (NumIncoming > MaxNEntries) {
            Entry = EntryCandidate;
            ShortestPath = mapAt(ShortestPathFromEntry, EntryCandidate);
          } else if (NumIncoming == MaxNEntries) {
            size_t SP = mapAt(ShortestPathFromEntry, EntryCandidate);
            if (SP < ShortestPath) {
              Entry = EntryCandidate;
              ShortestPath = SP;
            }
          }
        }
      }
    }
    assert(Entry != nullptr);
    return Entry;
  }

  template<class NodeT>
  bool setContains(llvm::SmallPtrSetImpl<NodeT> &Set, NodeT &Element) const {
    return Set.contains(Element);
  }

  template<class GraphT,
           class GT = llvm::GraphTraits<llvm::Inverse<GraphT>>,
           typename NodeRef = typename GT::NodeRef>
  llvm::SmallVector<std::pair<NodeRef, NodeRef>, 4>
  getOutlinedEntries(llvm::DenseMap<NodeRef, size_t> &EntryCandidates,
                     BlockSet &Region,
                     NodeRef Entry) const {
    llvm::SmallVector<std::pair<NodeRef, NodeRef>, 4> LateEntryPairs;
    for (const auto &[Other, NumIncoming] : EntryCandidates) {
      if (Other != Entry) {
        llvm::SmallVector<NodeRef, 4> OutsidePredecessor;
        for (NodeRef Predecessor : llvm::make_range(GT::child_begin(Other),
                                                    GT::child_end(Other))) {
          if (not setContains(Region, Predecessor)) {
            OutsidePredecessor.push_back(Predecessor);
            LateEntryPairs.push_back( {Predecessor, Other} );
          }
        }
        assert(OutsidePredecessor.size() == NumIncoming);

        // Print all the outside predecessor.
        llvm::dbgs() << "\nNon regular entry candidates found:\n";
        printVector(OutsidePredecessor);

      }
    }

    return LateEntryPairs;
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
  bool outlineFirstIteration(llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>,
                                               4> &LateEntryPairs,
                             BlockSet &Region,
                             BlockSet &OutlinedNodes,
                             mlir::Block *Entry,
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
        LateEntryPairs.push_back( {LateEntryClone, Successor });
      }
    }
  }

  return OutlinedCycle;
}

  void performRestructureCFRegion(mlir::Region &reg,
                                  mlir::PatternRewriter &rewriter) const {
    bool OutlinedCycle = true;
    while (OutlinedCycle) {
      OutlinedCycle = false;

      //reg.viewGraph();
      EdgeSet backedges = getBackedges(&reg.front());
      llvm::dbgs() << "\nInitial backedges:\n";
      printBackedges(backedges);
      llvm::dbgs() << "\nInitial reachables:\n";
      printReachableBlocks(backedges);

      // Declare a metaobject which will contain all the identified region
      // objects.
      BlockSetVect regions;

      // Push manually the `root` region in the identified regions.
      BlockSet RootRegion;
      for (mlir::Block &B : reg) {
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

      // Create the parent region tree.
      revng::detail::ParentTree<mlir::Block *> Pt;

      // Insert the regions in the region tree.
      for (BlockSet &Region : regions) {
        Pt.insertRegion(Region);
      }

      // Order the regions inside the `ParentTree`.
      Pt.order();

      // Order the regions so that they go from the outer one to the inner one.
      llvm::dbgs() << "\nAfter ordering:\n";
      printRegions(Pt.getRegions());

      llvm::SmallVector<mlir::Block *, 4> RPOT;
      using RPOTraversal = llvm::ReversePostOrderTraversal<mlir::Region *>;
      llvm::copy(RPOTraversal(&reg), std::back_inserter(RPOT));

      llvm::DenseMap<mlir::Block *, size_t> ShortestPathFromEntry = computeDistanceFromEntry(&reg);

      size_t RegionIndex = 0;
      for (BlockSet &region : Pt.regions()) {
        llvm::dbgs() << "\nRestructuring region idx: " << RegionIndex << ":\n";
        llvm::DenseMap<mlir::Block *, size_t> EntryCandidates = getEntryCandidates<mlir::Block *>(region);

        // In case we analyzing the root region, we expect to have no entry
        // candidates.
        bool RootRegionIteration = (RegionIndex + 1) == regions.size();
        assert(!EntryCandidates.empty() || RootRegionIteration);
        assert(!RootRegionIteration || EntryCandidates.empty());
        if (RootRegionIteration) {
          assert(EntryCandidates.empty());
          EntryCandidates.insert({ RPOT.front(), 0 });
        }

        llvm::dbgs() << "\nEntry candidates:\n";
        printMap(EntryCandidates);

        mlir::Block *Entry = electEntry<mlir::Block *>(EntryCandidates,
                                                      ShortestPathFromEntry,
                                                      RPOT);
        llvm::dbgs() << "\nElected entry:\n";
        Entry->printAsOperand(llvm::dbgs());
        llvm::dbgs() << "\n";

        // Now that elected the entry node we can prooced with the inlining.
        // Extract for each non-elected entry, the inlining path.
        llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>, 4> LateEntryPairs
          = getOutlinedEntries<mlir::Block *>(EntryCandidates, region, Entry);

        // Outline the first iteration of the cycles.
        BlockSet OutlinedNodes;
        OutlinedCycle = outlineFirstIteration(LateEntryPairs, region, OutlinedNodes, Entry, rewriter);

        // Add the outlined nodes to the current region.
        region.insert(OutlinedNodes.begin(), OutlinedNodes.end());

        // Re-add all the nodes of the current region to the parent region.
        if (Pt.hasParent(region)) {
          Pt.getParent(region).insert(region.begin(), region.end());
        }

        // Increment region index for next iteration.
        RegionIndex++;
      }
    }
  }
};

struct RestructureCFPass
    : public PassWrapper<RestructureCFPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RestructureCFPass)

  StringRef getArgument() const final { return "restructure-cf"; }
  StringRef getDescription() const final {
    return "Perform restructuring on `cf` dialect";
  }
  RestructureCFPass() = default;
  RestructureCFPass(const RestructureCFPass &) {}

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RestructureCFRewriter>(&getContext());

    SmallVector<Operation *> Functions;
    getOperation()->walk([&](LLVM::LLVMFuncOp F) {
      Functions.push_back(F);
    });

    auto Strictness = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyOpPatternsAndFold(Functions,
                                      std::move(patterns),
                                      Strictness)))
      signalPassFailure();
  }

  Option<bool> performRestructure{*this, "restructure",
                                  llvm::cl::desc("Restructure CF dialect"),
                                  llvm::cl::init(false)};
};
} // namespace

namespace mlir {
namespace test {
void registerRestructureCFPass() {
  PassRegistration<RestructureCFPass>();
}
} // namespace test
} // namespace mlir
