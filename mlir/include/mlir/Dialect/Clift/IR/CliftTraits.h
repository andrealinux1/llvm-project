#pragma once

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {

template <typename UseType>
struct OneUseOfType {
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, Impl> {
  private:
    using Base = TraitBase<ConcreteType, Impl>;

  public:
    static LogicalResult verifyTrait(Operation *op) {
      static_assert(ConcreteType::template hasTrait<OneResult>(),
                    "expected operation to produce one result");

      mlir::Value result = op->getResult(0);

      const size_t numUsesOfProvidedType =
          llvm::count_if(result.getUses(), [](mlir::OpOperand &operand) {
            return mlir::isa<UseType>(operand.getOwner());
          });

      if (numUsesOfProvidedType != 1)
        return op->emitOpError() << "expects to have a single use which is a "
                                 << UseType::getOperationName();
      return success();
    }
  };
};

} // namespace OpTrait
} // namespace mlir
