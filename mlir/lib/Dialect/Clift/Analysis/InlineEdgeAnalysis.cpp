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

template <class NodeT>
mlir::CliftInlinedEdge<NodeT>::CliftInlinedEdge(mlir::Region &Region,
                                                DominanceInfo &DomInfo) {

  // TODO: migrate this helper analysis class to use the MFP implementation once
  // the code is merged into orchestra dev environment.

  // Helpers sets to contain exit nodes reachability used in the "different
  // exits" analysis.
  llvm::DenseMap<NodeT *, llvm::SmallPtrSet<NodeT *, 4>> ReachableExits;
  for (NodeT &Exit : Region) {

    // Start the backward DFS only from exit nodes in the region.
    if (successor_range_size(&Exit) > 0) {
      continue;
    }

    // Mark each node reachable from an exit nodes as reachable from such exit.
    for (NodeT *DFSBlock : llvm::inverse_depth_first(&Exit)) {
      ReachableExits[DFSBlock].insert(&Exit);
    }
  }

  // Collect all the conditional nodes in the loop region.
  llvm::SmallVector<NodeT *> ConditionalBlocks;
  for (NodeT *B : llvm::post_order(&(Region.front()))) {

    // Enqueue all blocks with more than one successor as conditional nodes to
    // process.
    if (successor_range_size(B) >= 2) {
      ConditionalBlocks.push_back(B);
    }
  }

  // TODO: We considered using a mlir Dialect attribute to store the inlined
  // property on an edge. Unforunately this option can cause a significant
  // effort in maintaining the attribute update along the mlir pipeline. For
  // this reason, we implemented an analysis through the `CliftInlinedEdge`
  // class to compute and query such property.

  // Iterate over the conditional nodes, and carry out the separate reachable
  // exit analysis.
  for (NodeT *B : ConditionalBlocks) {
    if (successor_range_size(B) == 2) {
      NodeT *Then = get_successor(B, 0);
      NodeT *Else = get_successor(B, 1);
      auto ThenExits = ReachableExits[Then];
      auto ElseExits = ReachableExits[Else];

      llvm::SmallPtrSet<NodeT *, 4> Intersection = ThenExits;
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

      for (NodeT *Exit : ThenExits) {
        if (not DomInfo.dominates(B, Exit)) {
          ThenIsDominated = false;
          break;
        }
      }

      for (NodeT *Exit : ElseExits) {
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

// Explicit instantiation for the `mlir::Block *` class.
template class mlir::CliftInlinedEdge<mlir::Block>;
