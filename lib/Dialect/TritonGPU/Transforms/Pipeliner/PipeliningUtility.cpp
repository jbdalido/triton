#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

bool mlir::triton::loopHasDistGreaterThanOne(scf::ForOp forOp) {
  return llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                      [](Value operand) {
                        Operation *def = operand.getDefiningOp();
                        return !def;
                      });
}

bool mlir::triton::isOuterLoop(scf::ForOp forOp) {
  return llvm::any_of(forOp.getBody()->getOperations(), [](Operation &op) {
    return isa<scf::ForOp, scf::WhileOp>(op);
  });
}

// Combine the current mask with the given predicate.
static Value getPredMask(RewriterBase &rewriter, Type typeLike,
                         Value currentMask, Value pred) {
  Type maskType = tt::getI1SameShape(typeLike);
  Location loc = pred.getLoc();
  Value mask = pred;
  if (isa<RankedTensorType>(maskType)) {
    mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);
  }
  if (currentMask) {
    mask = rewriter.create<arith::AndIOp>(loc, mask, currentMask);
  }
  return mask;
}

// Function to mask operations during scheduling.
Operation *mlir::triton::predicateOp(RewriterBase &rewriter, Operation *op,
                                     Value pred) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (mlir::isMemoryEffectFree(op))
    return op;
  if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op))
    return op;
  if (isa<ttg::LocalLoadOp, ttg::LocalStoreOp>(op))
    return op;
  if (isa<ttng::TMEMAllocOp, ttng::TMEMCopyOp>(op))
    return op;
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    rewriter.setInsertionPoint(op);
    Value cnd = getPredMask(rewriter, ifOp.getCondition().getType(),
                            ifOp.getCondition(), pred);
    ifOp.getConditionMutable().assign(cnd);
    return op;
  }
  if (auto asyncCopyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
    rewriter.setInsertionPoint(asyncCopyOp);
    Value mask = getPredMask(rewriter, asyncCopyOp.getSrc().getType(),
                             asyncCopyOp.getMask(), pred);
    asyncCopyOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    rewriter.setInsertionPoint(loadOp);
    Value mask = getPredMask(rewriter, loadOp.getPtr().getType(),
                             loadOp.getMask(), pred);
    loadOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
    rewriter.setInsertionPoint(copyOp);
    Value mask = getPredMask(rewriter, copyOp.getPred().getType(),
                             copyOp.getPred(), pred);
    copyOp.getPredMutable().assign(mask);
    return op;
  }
  if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
    rewriter.setInsertionPoint(gatherOp);
    Value mask = getPredMask(rewriter, gatherOp.getPred().getType(),
                             gatherOp.getPred(), pred);
    gatherOp.getPredMutable().assign(mask);
    return op;
  }
  if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op)) {
    rewriter.setInsertionPoint(expectOp);
    Value mask = getPredMask(rewriter, expectOp.getPred().getType(),
                             expectOp.getPred(), pred);
    expectOp.getPredMutable().assign(mask);
    return op;
  }
  if (auto mmav5Op = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    rewriter.setInsertionPoint(mmav5Op);
    auto currPred = mmav5Op.getPredicate();
    Value mask = getPredMask(rewriter, currPred.getType(), currPred, pred);
    mmav5Op.setPredicate(mask);
    return op;
  }
  if (auto tmemStoreOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
    rewriter.setInsertionPoint(tmemStoreOp);
    Value mask = getPredMask(rewriter, tmemStoreOp.getPred().getType(),
                             tmemStoreOp.getPred(), pred);
    tmemStoreOp.getPredMutable().assign(mask);
    return op;
  }
  if (auto waitBarrier = dyn_cast<ttng::WaitBarrierOp>(op)) {
    rewriter.setInsertionPoint(waitBarrier);
    Value mask = pred;
    Value currentPred = waitBarrier.getPred();
    if (currentPred) {
      mask = getPredMask(rewriter, currentPred.getType(), currentPred, pred);
    }
    waitBarrier.getPredMutable().assign(mask);
    return op;
  }
  if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
    rewriter.setInsertionPoint(storeOp);
    Value mask = getPredMask(rewriter, storeOp.getPtr().getType(),
                             storeOp.getMask(), pred);
    storeOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto atomicRMWOp = dyn_cast<tt::AtomicRMWOp>(op)) {
    rewriter.setInsertionPoint(atomicRMWOp);
    Value mask = getPredMask(rewriter, atomicRMWOp.getPtr().getType(),
                             atomicRMWOp.getMask(), pred);
    atomicRMWOp.getMaskMutable().assign(mask);
    return op;
  }

  op->emitError("pipeliner doesn't know how to predicate this op.");
  llvm::report_fatal_error("Fatal pipeliner error");
  return op;
}

/// Helper to recursively add dependencies to the same stage.
void mlir::triton::addDep(Operation *op, DenseSet<Operation *> &deps,
                          bool includeArg, DenseSet<Operation *> *filter) {
  if (filter && filter->count(op))
    return;
  if (!deps.insert(op).second)
    return;
  for (Value operand : op->getOperands()) {
    Value v = operand;
    llvm::SmallDenseSet<Value> seen;
    while (auto arg = mlir::dyn_cast<BlockArgument>(v)) {
      if (!includeArg)
        break;
      if (!seen.insert(v).second)
        break;
      if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
        auto yieldOp = op->getBlock()->getTerminator();
        v = yieldOp->getOperand(arg.getArgNumber() - 1);
        continue;
      }
      break;
    }
    Operation *defOp = v.getDefiningOp();
    if (defOp && defOp->getBlock() == op->getBlock()) {
      addDep(defOp, deps, includeArg, filter);
    }
  }
}

// Add operations to the schedule with the given stage based on the filter
// function.
void mlir::triton::addOps(
    scf::ForOp forOp, int stage,
    std::vector<std::pair<Operation *, unsigned>> &schedule,
    std::function<bool(Operation *)> filter) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!filter(&op))
      continue;
    schedule.emplace_back(&op, stage);
  }
}

void mlir::triton::replaceUsesAndPropagateType(OpBuilder &builder,
                                               Operation *oldUse, Value val) {
  SmallVector<Operation *> opsToDelete;
  SmallVector<OpOperand *> operandsToReplace;

  // Save the operand to replace / delete later (avoid iterator invalidation).
  // TODO: can we use an early_inc iterator?
  for (OpOperand &use : oldUse->getUses()) {
    // Non-subview/trans ops will be replaced by `val`.
    if (!isa<triton::gpu::MemDescTransOp, triton::gpu::MemDescSubviewOp>(
            use.getOwner())) {
      operandsToReplace.push_back(&use);
      continue;
    }
    Operation *user = use.getOwner();
    // `subview(old_op)` is replaced by a new `subview(val)`.
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(user);
    Value newVal;
    if (auto subview = dyn_cast<triton::gpu::MemDescSubviewOp>(user)) {
      triton::gpu::MemDescType oldType = subview.getType();
      bool isMutable =
          cast<triton::gpu::MemDescType>(val.getType()).getMutableMemory();
      Type newDstType = triton::gpu::MemDescType::get(
          oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
          oldType.getMemorySpace(), isMutable);
      newVal = builder.create<triton::gpu::MemDescSubviewOp>(
          subview.getLoc(), newDstType, val, subview.getOffsets());
      newVal.getDefiningOp()->setAttrs(user->getAttrs());
    } else if (auto trans = dyn_cast<triton::gpu::MemDescTransOp>(user)) {
      newVal = builder.create<triton::gpu::MemDescTransOp>(trans.getLoc(), val,
                                                           trans.getOrder());
      newVal.getDefiningOp()->setAttrs(user->getAttrs());
    }
    assert(newVal);
    replaceUsesAndPropagateType(builder, user, newVal);
    opsToDelete.push_back(use.getOwner());
  }

  // Perform late replacement.
  for (OpOperand *operand : operandsToReplace) {
    Operation *op = operand->getOwner();
    operand->set(val);
  }

  // Perform late op erasure.
  for (Operation *op : opsToDelete)
    op->erase();
}

// Return true if the given ForOp has the attribute
// `tt.disallow_acc_multi_buffer` set to true.
bool mlir::triton::getDisallowAccMultiBuffer(scf::ForOp forOp) {
  return forOp->hasAttr(mlir::triton::kDisallowAccMultiBufferAttrName);
}

std::optional<std::pair<int, int>>
mlir::triton::maybeGetStageCluster(Operation *op) {
  auto stage =
      dyn_cast_if_present<IntegerAttr>(op->getAttr(tt::kLoopStageAttrName));
  auto clusterId =
      dyn_cast_if_present<IntegerAttr>(op->getAttr(tt::kLoopClusterAttrName));
  if (!stage || !clusterId) {
    return std::nullopt;
  }

  return {
      {stage.getValue().getSExtValue(), clusterId.getValue().getSExtValue()}};
}
std::pair<int, int> mlir::triton::getStageCluster(Operation *op) {
  auto res = maybeGetStageCluster(op);
  assert(res.has_value() && "Operation is missing stage & cluster attribute");
  return *res;
}

void mlir::triton::setStageCluster(Operation *op, int stage, int cluster) {
  auto ctx = op->getContext();
  op->setAttr(mlir::triton::kLoopStageAttrName,
              IntegerAttr::get(IntegerType::get(ctx, 32), stage));
  op->setAttr(mlir::triton::kLoopClusterAttrName,
              IntegerAttr::get(IntegerType::get(ctx, 32), cluster));
}

std::pair<int, int> mlir::triton::getMinMaxCluster(scf::ForOp &forOp) {
  int minClusterId = -1, maxClusterId = -1;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (!op.hasAttr(mlir::triton::kLoopStageAttrName) ||
        !op.hasAttr(mlir::triton::kLoopClusterAttrName))
      continue;
    auto [_, cluster] = getStageCluster(&op);
    if (maxClusterId < 0) {
      minClusterId = cluster;
      maxClusterId = cluster;
      continue;
    }
    maxClusterId = cluster > maxClusterId ? cluster : maxClusterId;
    minClusterId = cluster < minClusterId ? cluster : minClusterId;
  }
  return std::make_pair(minClusterId, maxClusterId);
}
