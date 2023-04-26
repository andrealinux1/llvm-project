//===- InlineEdgeAnalysis.cpp - Dependence analysis on SSA views ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the inline edge analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Clift/Analysis/InlineEdgeAnalysis.h"
#include "mlir/Dialect/Clift/IR/Clift.h"
#include "mlir/Dialect/Clift/IR/CliftDebug.h"
#include "mlir/GraphAlgorithms/GraphAlgorithms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/RegionGraphTraits.h"

#include "llvm/Support/Debug.h"

template <class GraphT>
mlir::CliftInlinedEdge<GraphT>::CliftInlinedEdge(mlir::Region &Region,
                                                 DominanceInfo &DomInfo,
                                                 PostDominanceInfo &PostDomInfo)
    : DomInfo(DomInfo), PostDomInfo(PostDomInfo) {

  // TODO: migrate this helper analysis class to use the MFP implementation once
  // the code is merged into orchestra dev environment.

  // Helpers sets to contain exit nodes reachability used in the "different
  // exits" analysis.
  llvm::DenseMap<mlir::Block *, llvm::SmallPtrSet<mlir::Block *, 4>>
      ReachableExits;
  for (mlir::Block &Exit : Region) {

    // Start the backward DFS only from exit nodes in the region.
    if (successor_range_size(&Exit) > 0) {
      continue;
    }

    // Mark each node reachable from an exit nodes as reachable from such exit.
    for (mlir::Block *DFSBlock : llvm::inverse_depth_first(&Exit)) {
      ReachableExits[DFSBlock].insert(&Exit);
    }
  }

  // Collect all the conditional nodes in the loop region.
  llvm::SmallVector<mlir::Block *> ConditionalBlocks;
  for (mlir::Block *B : llvm::post_order(&(Region.front()))) {

    // Enqueue all blocks with more than one successor as conditional nodes to
    // process.
    if (successor_range_size(B) >= 2) {
      ConditionalBlocks.push_back(B);
    }
  }

  // Iterate over the conditional nodes, and carry out the separate reachable
  // exit analysis.

  // TODO: at the present time, we make use of a set to contain and mark the
  // edges that we consider as inlined. The Correct Way TM to do this, is to
  // make use of Dialect attributes in order to specify that a certain
  // successor is inlined. Unfortunately, we can only define attributes for
  // our own dialect, so we first need to implement the control flow
  // operations for Clift.
  for (mlir::Block *B : ConditionalBlocks) {
    if (successor_range_size(B) == 2) {

      // TODO: migrate this to the use of traits, and not mlir::Block
      // methods.
      mlir::Block *Then = B->getSuccessor(0);
      mlir::Block *Else = B->getSuccessor(1);
      auto ThenExits = ReachableExits[Then];
      auto ElseExits = ReachableExits[Else];

      llvm::SmallPtrSet<mlir::Block *, 4> Intersection = ThenExits;
      llvm::set_intersect(Intersection, ElseExits);

      // If the exit sets are disjoint, we can avoid processing the
      // conditional node in the comb operation.
      if (not Intersection.empty()) {
        continue;
      }

      // Further check that we do not dominate at maximum one of the two set
      // of reachable exits.
      bool ThenIsDominated = true;
      bool ElseIsDominated = true;

      for (mlir::Block *Exit : ThenExits) {
        if (not DomInfo.dominates(B, Exit)) {
          ThenIsDominated = false;
          break;
        }
      }

      for (mlir::Block *Exit : ElseExits) {
        if (not DomInfo.dominates(B, Exit)) {
          ElseIsDominated = false;
          break;
        }
      }

      // If there is a set of exits that the current conditional block
      // entirely dominates, we can blacklist it because it will never cause
      // duplication. The reason is that the set of exits that we dominate can
      // be compltetely inlined and absorbed either into the `then` or into
      // the `else`.
      if (ThenIsDominated or ElseIsDominated) {

        // We may be in the following situations:
        // 1. Both the `then` and `else` branches are dominated by the current
        // conditional block.
        if (ThenIsDominated and ElseIsDominated) {
          IE.insert(EdgeDescriptor(B, Then));
          IE.insert(EdgeDescriptor(B, Else));

          // Debug output.
          llvm::dbgs() << "Inlining edge: ";
          printEdge(EdgeDescriptor(B, Then));
          llvm::dbgs() << "Inlining edge: ";
          printEdge(EdgeDescriptor(B, Else));
        } else if (ThenIsDominated) {

          // 2. Only the `then` branch is dominated by the current conditional.
          IE.insert(EdgeDescriptor(B, Then));

          // Debug output.
          llvm::dbgs() << "Inlining edge: ";
          printEdge(EdgeDescriptor(B, Then));
        } else if (ElseIsDominated) {

          // 3. Only the `else` branch is dominated by the current conditional.
          IE.insert(EdgeDescriptor(B, Else));

          // Debug output.
          llvm::dbgs() << "Inlining edge: ";
          printEdge(EdgeDescriptor(B, Else));
        } else {
          std::abort();
        }
      }
    }
  }
}

// Explicit instantiation of template `CliftInlinedEdge` class.
template class mlir::CliftInlinedEdge<mlir::Block *>;
