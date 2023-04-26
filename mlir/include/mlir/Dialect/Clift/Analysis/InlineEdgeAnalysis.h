//===- InlineEdgeAnalysis.h - Inline Edge Analysis --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CLIFT_ANALYSIS_INLINEDGEANALYSIS_H_
#define MLIR_DIALECT_CLIFT_ANALYSIS_INLINEDGEANALYSIS_H_

#include "mlir/Dialect/Clift/IR/Clift.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/SmallSet.h"

namespace mlir {

class InlinedEdges {
public:
  using EdgeDescriptor = std::pair<mlir::Block *, mlir::Block *>;
  using EdgeSet = llvm::SmallSet<EdgeDescriptor, 4>;

public:
  InlinedEdges() { InlinedEdgesSet.clear(); }

  void insert(EdgeDescriptor Edge) { InlinedEdgesSet.insert(Edge); }
  bool isInlined(EdgeDescriptor Edge) { return InlinedEdgesSet.count(Edge); }

private:
  EdgeSet InlinedEdgesSet;
};

template <class GraphT>
class CliftInlinedEdge {
  using EdgeDescriptor = InlinedEdges::EdgeDescriptor;

public:
  CliftInlinedEdge(mlir::Region &Region, DominanceInfo &DomInfo,
                   PostDominanceInfo &PostDomInfo);

  bool isInlined(EdgeDescriptor Edge) { return IE.isInlined(Edge); }

private:
  InlinedEdges IE;
  DominanceInfo &DomInfo;
  PostDominanceInfo &PostDomInfo;
};

} // namespace mlir

#endif
