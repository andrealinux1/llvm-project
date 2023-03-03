
#include <llvm/Support/Debug.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

static void swapThenElseBranches(scf::IfOp If, PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(If);
  Value xoruno = rewriter.create<arith::XOrIOp>(If.getLoc(),
                                                If.getCondition().getType(),
                                                If.getCondition(),
                                                rewriter.create<arith::ConstantOp>(If.getLoc(), rewriter.getBoolAttr(1)));
  auto NewIf = rewriter.create<scf::IfOp>(If.getLoc(), If.getResults().getTypes(), xoruno, true);
  rewriter.cloneRegionBefore(If.getThenRegion(), &NewIf.getElseRegion().front());
  rewriter.eraseBlock(&NewIf.getElseRegion().back());
  rewriter.cloneRegionBefore(If.getElseRegion(), &NewIf.getThenRegion().front());
  rewriter.eraseBlock(&NewIf.getThenRegion().back());
  rewriter.replaceOp(If, NewIf.getResults());
}

class TransformationPassRewriter : public OpRewritePattern<scf::IfOp> {
  using mlir::OpRewritePattern<scf::IfOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(scf::IfOp op,
                                      mlir::PatternRewriter &rewriter) const final {
    swapThenElseBranches(op, rewriter);
    return success();
  }
};

struct TestTransformationPass
    : public PassWrapper<TestTransformationPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTransformationPass)

  StringRef getArgument() const final { return "test-transformation"; }
  StringRef getDescription() const final {
    return "Tests customary SCF transformations";
  }
  TestTransformationPass() = default;
  TestTransformationPass(const TestTransformationPass &) {}

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TransformationPassRewriter>(&getContext());

    SmallVector<scf::IfOp, 4> Ifs;
    getOperation()->walk([&](scf::IfOp If) {
      Ifs.push_back(If);
    });
    for (scf::IfOp If : Ifs) {
      assert(If.getNumRegions() == 2);
      if (failed(applyOpPatternsAndFold(If, std::move(patterns))))
        signalPassFailure();
    }
  }

  void printOperation(Operation *Op) {
    // We print the operation we are currently visiting.
    printIndent() << "visiting op: '" << Op->getName() << "'\n";
    printIndent() << "It has '" << Op->getNumOperands() << "' operands and '"
                  << Op->getNumResults() << " results'\n";

    // We now move to dumping the nested regions.
    printIndent() << "There are '" << Op->getNumRegions() << "' nested regions\n";
    auto indent = pushIndent();
    for (Region &NestedRegion : Op->getRegions()) {
      //printRegion(NestedRegion);
      //printIndent() << NestedRegion.getName() << "\n";
      NestedRegion.viewGraph();
    }
  }

  Option<bool> flipElse{*this, "flipelse",
                        llvm::cl::desc("Flip the else branch of a scf.if"),
                        llvm::cl::init(false)};

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };
  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      llvm::outs() << "  ";
    return llvm::outs();
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestTransformationPass() {
  PassRegistration<TestTransformationPass>();
}
} // namespace test
} // namespace mlir
