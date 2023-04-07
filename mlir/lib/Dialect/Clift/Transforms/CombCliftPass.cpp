//===- CombCliftPass.cpp - perform restructuring on Clift -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Performs comb on Clift.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Clift/Transforms/Passes.h"

#include "mlir/Dialect/Clift/IR/Clift.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_COMBCLIFT
#include "mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

class CombCliftRewriter : public OpRewritePattern<LLVM::LLVMFuncOp> {

  using mlir::OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp Op,
                  mlir::PatternRewriter &Rewriter) const final {

    // Ensure that we start from a `LLVMFuncOp` with a single `cf` region.
    assert(Op->getNumRegions() == 1);

    // Transform only regions which have an actual size.
    mlir::Region &FunctionRegion = Op->getRegion(0);
    if (not FunctionRegion.getBlocks().empty()) {
      performCombCliftRegion(FunctionRegion, Rewriter);
    }
    return success();
  }

  void performCombCliftRegion(mlir::Region &FunctionRegion,
                              mlir::PatternRewriter &Rewriter) const {

    // TODO: implement implementation here.
  }
};

struct CombClift : public impl::CombCliftBase<CombClift> {
  void runOnOperation() override {
    RewritePatternSet Patterns(&getContext());
    Patterns.add<CombCliftRewriter>(&getContext());

    SmallVector<Operation *> Functions;
    getOperation()->walk([&](LLVM::LLVMFuncOp F) { Functions.push_back(F); });

    auto Strictness = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(
            applyOpPatternsAndFold(Functions, std::move(Patterns), Strictness)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createCombCliftPass() {
  return std::make_unique<CombClift>();
}
