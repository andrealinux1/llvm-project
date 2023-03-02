add_mlir_unittest(MLIRCliftTests
  CliftTest.cpp
)
target_link_libraries(MLIRCliftTests
  PRIVATE
  MLIRCliftDialect
  )
