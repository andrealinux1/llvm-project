//===- RegionTreeGraphTraits.h - GraphTraits for RegionTree -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements specializations of llvm::GraphTraits for the
// `RegionNode` and `RegionTree` classes. This allows the generic LLVM graph
// algorithms to be applied to the RegionTree hierarchy.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REGIONTREEGRAPHTRAITS_H
#define MLIR_IR_REGIONTREEGRAPHTRAITS_H

#include "mlir/GraphAlgorithms/GraphAlgorithms.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/GraphTraits.h"

namespace llvm {
template <>
struct GraphTraits<revng::detail::RegionNode<mlir::Block *> *> {
  using Node = revng::detail::RegionNode<mlir::Block *>;
  using ChildIteratorType = decltype(std::declval<Node>().succ_begin());

  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef BB) { return BB; }

  static ChildIteratorType child_begin(NodeRef Node) {
    return Node->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef Node) { return Node->succ_end(); }
};

template <>
struct GraphTraits<revng::detail::RegionTree<mlir::Block *> *>
    : public GraphTraits<revng::detail::RegionNode<mlir::Block *> *> {
  using GraphType = revng::detail::RegionTree<mlir::Block *> *;
  using NodeRef = revng::detail::RegionNode<mlir::Block *> *;

  static NodeRef getEntryNode(GraphType Rt) { return &Rt->front(); }

  using nodes_iterator = revng::detail::RegionTree<mlir::Block *>::links_it;
  static nodes_iterator nodes_begin(GraphType Rt) {
    return nodes_iterator(Rt->begin());
  }
  static nodes_iterator nodes_end(GraphType Rt) {
    return nodes_iterator(Rt->end());
  }
};

} // namespace llvm

#endif
