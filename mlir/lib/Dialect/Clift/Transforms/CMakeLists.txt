add_mlir_dialect_library(MLIRCliftTransforms
  CombCliftPass.cpp
  RestructureCliftPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Clift

  DEPENDS
  MLIRCliftPassIncGen

  LINK_LIBS PUBLIC
  MLIRCliftDialect
  MLIRDialect
  MLIRIR
  )
