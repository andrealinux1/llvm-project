#include "mlir/Dialect/Clift/IR/Clift.h"
#include "mlir/Dialect/Clift/IR/CliftTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/Clift/IR/CliftOpsDialect.cpp.inc"

class TypeAliasASMInterface : public mlir::OpAsmDialectInterface {
public:
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Type type, llvm::raw_ostream &OS) const final {

    return AliasResult::NoAlias;
  }
};

void mlir::clift::CliftDialect::initialize() {
  registerTypes();
  registerOperations();
  addInterfaces<TypeAliasASMInterface>();
}
