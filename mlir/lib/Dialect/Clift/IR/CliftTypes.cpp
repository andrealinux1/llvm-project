#include "mlir/Dialect/Clift/IR/CliftTypes.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"

void mlir::clift::CliftDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"
      >();
}

/// Parse a type registered to this dialect.
::mlir::Type
mlir::clift::CliftDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;

  parser.emitError(typeLoc) << "unknown  type `" << mnemonic << "` in dialect `"
                            << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void mlir::clift::CliftDialect::printType(
    ::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
}
