#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/GraphAlgorithms/GraphAlgorithms.h"

#include "llvm/Support/Debug.h"

using EdgeDescriptor = revng::detail::EdgeDescriptor<mlir::Block *>;
using EdgeSet = llvm::SmallSetVector<EdgeDescriptor, 4>;
using BlockSet = llvm::SmallSetVector<mlir::Block *, 4>;
using BlockSetVect = llvm::SmallVector<BlockSet>;

using RegionNode = revng::detail::RegionNode<mlir::Block *>;
using BlockNode = RegionNode::BlockNode;
using ChildRegionDescriptor = RegionNode::ChildRegionDescriptor;
using RegionTree = revng::detail::RegionTree<mlir::Block *>;

void printBlock(mlir::Block *Block);

void printChildRegionDescriptorIndex(ChildRegionDescriptor &ChildRegion);

void printBlockOrIndex(
    const std::variant<mlir::Block *, ChildRegionDescriptor *> &Node);

void printEdge(const EdgeDescriptor &Edge);

void printBackedge(EdgeDescriptor &Backedge);

void printBackedges(EdgeSet &Backedges);

void printReachableBlocks(EdgeSet &Backedges);

void printRegions(BlockSetVect &Regions);

void printMap(SmallDenseMap<mlir::Block *, size_t> &Map);

void printVector(llvm::SmallVectorImpl<mlir::Block *> &Vector);

void printPairVector(
    llvm::SmallVectorImpl<std::pair<mlir::Block *, mlir::Block *>> &Vector);

void printRegionNode(RegionNode &RegionNode);

void printRegionTree(RegionTree &RegionTree);
