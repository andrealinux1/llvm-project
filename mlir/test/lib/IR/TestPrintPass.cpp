#include "llvm/Support/ScopedPrinter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct PrintPass : public PassWrapper<PrintPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintPass)

  StringRef getArgument() const final { return "test-print-pass"; }
  StringRef getDescription() const final { return "Custom test print."; }
  void runOnOperation() override {
    // Get the current FuncOp operation being operated on.
    Operation *Op = getOperation();
    resetIndent();
    //assert(false);
    printOperation(Op);
  }

  void printOperation(Operation *Op) {
  // We print the operation we are currently visiting.
  printIndent() << "visiting op: '" << Op->getName() << "'\n";
  printIndent() << "It has '" << Op->getNumOperands() << "' operands and '"
                << Op->getNumResults() << " results'\n";

  // We now move to dumping the nested regions.
  printIndent() << "There are '" << Op->getNumRegions() << "' nested regions\n";
  auto indent = pushIndent();
  for (Region &NestedRegion : Op->getRegions())
  printRegion(NestedRegion);
  }

  void printRegion(Region &NestedRegion) {
    printIndent() << "Region with " << NestedRegion.getBlocks().size()
                  << " blocks.\n";
    auto indent = pushIndent();
    for (Block &NestedBlock : NestedRegion.getBlocks())
      printBlock(NestedBlock);
  }

  void printBlock(Block &NestedBlock) {
    printIndent() << "Block with " << NestedBlock.getNumArguments()
                  << " arguments, " << NestedBlock.getNumSuccessors()
                  << " successors, and " << NestedBlock.getOperations().size()
                  << " operations\n";

    // Now the recursive step, which print all the operations hold by a block.
    auto indent = pushIndent();
    for (Operation &Op : NestedBlock.getOperations())
      printOperation(&Op);
  }

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
} // end anonymous namespace

namespace mlir {
void registerTestPrintPass() {
  PassRegistration<PrintPass>();
}

} //namespace mlir