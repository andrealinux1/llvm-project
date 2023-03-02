//===- InferShapeTest.cpp - unit tests for shape inference ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Clift/IR/Clift.h"
#include "mlir/Dialect/Clift/IR/CliftOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "gtest/gtest.h"

class CliftTest : public ::testing::Test {
public:
  CliftTest()
      : module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&context))),
        builder(module.getBodyRegion()) {

    registry.insert<mlir::clift::CliftDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  void canonicalize() {
    mlir::PassManager manager(&context);
    manager.addPass(mlir::createCanonicalizerPass());
    ASSERT_TRUE(manager.run(module).succeeded());
  }

protected:
  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
  std::unique_ptr<mlir::Diagnostic> diagnostic;
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
};

TEST_F(CliftTest, labelsWithoutGoToMustBeTriviallyDead) {
  auto label =
      builder.create<mlir::clift::MakeLabelOp>(builder.getUnknownLoc());
  builder.create<mlir::clift::AssignLabelOp>(builder.getUnknownLoc(), label);

  ASSERT_FALSE(module.getBody()->getOperations().empty());

  canonicalize();
  ASSERT_TRUE(module.getBody()->getOperations().empty());
}

TEST_F(CliftTest, labelsWithGoToMustBeAlive) {
  auto label =
      builder.create<mlir::clift::MakeLabelOp>(builder.getUnknownLoc());
  builder.create<mlir::clift::AssignLabelOp>(builder.getUnknownLoc(), label);
  builder.create<mlir::clift::GoToOp>(builder.getUnknownLoc(), label);

  ASSERT_FALSE(module.getBody()->getOperations().empty());

  canonicalize();
  ASSERT_EQ(module.getBody()->getOperations().size(), 3);
}

TEST_F(CliftTest, labelsWithAGoToWithoutAssignMustFail) {
  auto label =
      builder.create<mlir::clift::MakeLabelOp>(builder.getUnknownLoc());
  builder.create<mlir::clift::GoToOp>(builder.getUnknownLoc(), label);

  ASSERT_TRUE(mlir::verify(module).failed());
}
