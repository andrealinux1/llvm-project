#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/GraphAlgorithms/GraphAlgorithms.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/Support/Debug.h"

using EdgeDescriptor = revng::detail::EdgeDescriptor<mlir::Block *>;
using EdgeSet = llvm::SmallSetVector<EdgeDescriptor, 4>;
using BlockSet = llvm::SmallSetVector<mlir::Block *, 4>;
using BlockSetVect = llvm::SmallVector<BlockSet>;

using RegionNode = revng::detail::RegionNode<mlir::Block *>;
using BlockNode = RegionNode::BlockNode;
using ChildRegionDescriptor = RegionNode::ChildRegionDescriptor;
using RegionTree = revng::detail::RegionTree<mlir::Block *>;

void printBlock(mlir::Block *Block) { Block->printAsOperand(llvm::dbgs()); }

void printChildRegionDescriptorIndex(ChildRegionDescriptor &ChildRegion) {
  llvm::dbgs() << "Subregion ID: " << ChildRegion.ChildIndex << "\n";
}

void printBlockOrIndex(
    const std::variant<mlir::Block *, ChildRegionDescriptor *> &Node) {
  if (std::holds_alternative<mlir::Block *>(Node)) {
    mlir::Block *Block = std::get<mlir::Block *>(Node);
    printBlock(Block);
  } else if (std::holds_alternative<ChildRegionDescriptor *>(Node)) {
    ChildRegionDescriptor *ChildRegion =
        std::get<ChildRegionDescriptor *>(Node);
    printChildRegionDescriptorIndex(*ChildRegion);
  }
}

void printBackedge(EdgeDescriptor &Backedge) {
  llvm::dbgs() << "Backedge: ";
  printBlock(Backedge.first);
  llvm::dbgs() << " -> ";
  printBlock(Backedge.second);
  llvm::dbgs() << "\n";
}

void printBackedges(EdgeSet &Backedges) {
  for (EdgeDescriptor Backedge : Backedges) {
    printBackedge(Backedge);
  }
}

void printReachableBlocks(EdgeSet &Backedges) {
  for (EdgeDescriptor Backedge : Backedges) {
    printBackedge(Backedge);
    llvm::dbgs() << "We can reach blocks:\n";
    for (mlir::Block *Reachable :
         nodesBetween(Backedge.second, Backedge.first)) {
      printBlock(Reachable);
      llvm::dbgs() << "\n";
    }
  }
}

void printRegions(BlockSetVect &Regions) {
  size_t Regionindex = 0;
  for (BlockSet &Region : Regions) {
    llvm::dbgs() << "Region idx: " << Regionindex << " composed by nodes: \n";
    for (mlir::Block *Block : Region) {
      printBlock(Block);
      llvm::dbgs() << "\n";
    }
    Regionindex++;
  }
}

void printMap(SmallDenseMap<mlir::Block *, size_t> &Map) {
  llvm::dbgs() << "Map content:\n";
  for (auto const &[K, V] : Map) {
    printBlock(K);
    llvm::dbgs() << " -> " << V << "\n";
  }
}

void printVector(llvm::SmallVectorImpl<mlir::Block *> &Vector) {
  for (mlir::Block *Element : Vector) {
    printBlock(Element);
    llvm::dbgs() << "\n";
  }
}

void printPairVector(
    llvm::SmallVectorImpl<std::pair<mlir::Block *, mlir::Block *>> &Vector) {
  for (auto const &[First, Second] : Vector) {
    printBlock(First);
    llvm::dbgs() << " -> ";
    printBlock(Second);
    llvm::dbgs() << "\n";
  }
}

void printRegionNode(RegionNode &RegionNode) {
  for (BlockNode &Block : RegionNode) {
    printBlock(Block);
    llvm::dbgs() << "\n";
  }
  for (ChildRegionDescriptor &ChildRegion :
       RegionNode.successor_range_naked()) {
    printChildRegionDescriptorIndex(ChildRegion);
  }
}

void printRegionTree(RegionTree &RegionTree) {
  llvm::dbgs() << "\nRegionTree:\n";
  size_t RegionIndex = 0;
  for (RegionNode &RegionNode : RegionTree.regions()) {
    llvm::dbgs() << "Region idx: " << RegionIndex << " composed by nodes:\n";
    printRegionNode(RegionNode);
    RegionIndex++;
  }
}
