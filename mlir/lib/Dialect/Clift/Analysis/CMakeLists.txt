add_mlir_dialect_library(MLIRCliftAnalysis
  InlineEdgeAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Clift

  DEPENDS
  MLIRCliftPassIncGen

  LINK_LIBS PUBLIC
  MLIRCliftDialect
  MLIRDialect
  MLIRIR
  )
