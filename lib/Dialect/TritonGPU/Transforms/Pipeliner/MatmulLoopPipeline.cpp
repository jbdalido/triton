#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-matmul-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// TODO: We can extract some helpers into common utilities once we add more
// schedules.

namespace {

struct LoadInfo {
  // Layout of the data in shared memory.
  ttg::SharedEncodingTrait sharedEncoding = nullptr;
  // Blocked encoding is used for loads not used by the dot.
  ttg::BlockedEncodingAttr blockedEncoding = nullptr;
  bool isMMAv3Shared = false;
  bool isMMAv5Scale = false;
  int distToUse = 0;
  bool usedByDot = false;
};

} // namespace

class OpBuilderWithStage : public OpBuilder {
public:
  explicit OpBuilderWithStage(Operation *op,
                              OpBuilder::Listener *listener = nullptr)
      : OpBuilder(op, listener) {}
  explicit OpBuilderWithStage(Region &region, Listener *listener = nullptr)
      : OpBuilder(region, listener) {}

  template <typename OpTy, typename... Args>
  OpTy createWithStage(Location location, int stage, int cluster,
                       Args &&...args) {
    OpTy op = OpBuilder::create<OpTy>(location, std::forward<Args>(args)...);
    tt::setStageCluster(op, stage, cluster);
    return op;
  }
  using OpBuilder::create;
};

class OpBuilderForStage : public OpBuilder {
  std::optional<int> stage_, cluster_;

public:
  explicit OpBuilderForStage(Operation *op, int stage, int cluster)
      : OpBuilder(op, nullptr), stage_(stage), cluster_(cluster) {}
  explicit OpBuilderForStage(Operation *op) : OpBuilder(op, nullptr) {
    auto sc = tt::maybeGetStageCluster(op);
    if (sc) {
      stage_ = sc->first;
      cluster_ = sc->second;
    }
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    OpTy op = OpBuilder::create<OpTy>(std::forward<Args>(args)...);

    if (stage_ && cluster_) {
      tt::setStageCluster(op, *stage_, *cluster_);
    }
    return op;
  }
};

static bool sameStageCluster(Operation *op1, Operation *op2) {
  auto [s1, c1] = tt::getStageCluster(op1);
  auto [s2, c2] = tt::getStageCluster(op2);
  return s1 == s2 && c1 == c2;
}

// Return user of a loadOp with the lowest stage, if two users have the
// same stage, return the user with lower cluster.
static Operation *getFirstUseOfPipelinedLoad(Operation *loadOp) {
  Operation *firstUser = nullptr;
  for (Operation *user : loadOp->getUsers()) {
    if (user->getBlock() == loadOp->getBlock()) {
      auto [stage, clusterId] = tt::getStageCluster(user);
      // Update FirstUse if this use has lower stage or lower cluster.
      if (!firstUser)
        firstUser = user;
      else {
        auto [stageForFirstUse, clusterForFirstUse] =
            tt::getStageCluster(firstUser);
        if (stage < stageForFirstUse ||
            (stage == stageForFirstUse && clusterId < clusterForFirstUse))
          firstUser = user;
      }
    }
  }
  return firstUser;
}

static int createAsyncCopy(scf::ForOp forOp, tt::LoadOp loadOp, Value alloc,
                           Value insertIdx, Value extractIdx,
                           llvm::MapVector<Operation *, LoadInfo> &loadToInfo,
                           int maxClusterId) {
  int retCode = -1;
  OpBuilderWithStage builder(forOp);
  auto opPair = tt::getStageCluster(loadOp);
  auto *firstUse = getFirstUseOfPipelinedLoad(loadOp);
  auto [stageForFirstUse, clusterForFirstUse] = tt::getStageCluster(firstUse);
  int stage = opPair.first, clusterId = opPair.second;

  Value zero = builder.createWithStage<arith::ConstantIntOp>(
      forOp.getLoc(), stage, clusterId, 0, 32);

  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();
  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());

  auto convertBlockLayout = [&](Value src, ttg::BlockedEncodingAttr enc) {
    auto ty = cast<RankedTensorType>(src.getType());
    auto newTy = RankedTensorType::get(ty.getShape(), ty.getElementType(), enc);
    auto cvt = builder.createWithStage<ttg::ConvertLayoutOp>(
        loadOp->getLoc(), stage, clusterId, newTy, src);
    return cvt.getResult();
  };

  if (!isExpensiveLoadOrStore(loadOp) && loadToInfo[loadOp].blockedEncoding) {
    // For inexpensive loads that do not directly feed into dot ops
    // we want to use optimal layout for the data.
    ttg::BlockedEncodingAttr encoding = loadToInfo[loadOp].blockedEncoding;
    src = convertBlockLayout(src, encoding);
    if (mask)
      mask = convertBlockLayout(mask, encoding);
    if (other)
      other = convertBlockLayout(other, encoding);
  }

  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  ttg::MemDescType subviewTy = ttg::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true,
      /*allocShape=*/allocTy.getAllocShape());
  auto view = builder.createWithStage<ttg::MemDescSubviewOp>(
      loc, stage, clusterId, subviewTy, alloc, copyOffsets);
  Operation *copy = builder.createWithStage<ttg::AsyncCopyGlobalToLocalOp>(
      loc, stage, clusterId, src, view, mask, other, loadOp.getCache(),
      loadOp.getEvict(), loadOp.getIsVolatile());
  Operation *commit = builder.createWithStage<ttg::AsyncCommitGroupOp>(
      loc, stage, clusterId, copy->getResult(0));
  Operation *wait = builder.createWithStage<ttg::AsyncWaitOp>(
      loc, stageForFirstUse, clusterForFirstUse, commit->getResult(0), 0);

  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad = builder.createWithStage<ttg::MemDescSubviewOp>(
      loc, stageForFirstUse, clusterForFirstUse, subviewTy, alloc, loadOffsets);

  if (loadToInfo[loadOp].isMMAv3Shared || loadToInfo[loadOp].isMMAv5Scale) {
    auto user = *loadOp->getUsers().begin();
    assert(isa<triton::gpu::LocalAllocOp>(user) &&
           "Loading of MMAv3 operands and MMAv5 scale is expected to be "
           "consumed by LocalAlloc.");
    auto alloc = cast<ttg::LocalAllocOp>(user);
    tt::replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
    alloc.erase();
  } else {
    SmallVector<ttg::LocalAllocOp> allocsToErase;
    for (Operation *user : loadOp->getUsers()) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        tt::replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
        allocsToErase.push_back(alloc);
      }
    }
    for (auto alloc : allocsToErase) {
      alloc.erase();
    }

    auto sharedLoad = builder.createWithStage<ttg::LocalLoadOp>(
        loc, stageForFirstUse, clusterForFirstUse, loadOp.getType(), viewLoad,
        wait->getResult(0));
    auto result = sharedLoad->getResults();

    // Create a select for non-zero other values as they are not handled by
    // AsyncCopyGlobalToLocalOp for now.
    Value other = loadOp.getOther();
    if (other && !isZeroConst(other)) {
      auto select = builder.createWithStage<arith::SelectOp>(
          loc, stageForFirstUse, clusterForFirstUse, loadOp.getType(),
          // Use the mask operand from the original load, not the one with a
          // potentially transformed layout.
          loadOp.getMask(), sharedLoad.getResult(), other);
      result = select->getResults();
    }

    loadOp->replaceAllUsesWith(result);

    // Prefetch load if is not MMAV3 and is used by the dot.
    if (loadToInfo[loadOp].usedByDot) {
      assert(stageForFirstUse >= 1);
      tt::setStageCluster(wait, stageForFirstUse - 1, maxClusterId + 1);
      tt::setStageCluster(viewLoad, stageForFirstUse - 1, maxClusterId + 1);
      retCode = stageForFirstUse - 1;
    }
  }
  loadOp.erase();
  return retCode;
}

static void
createTMAAsyncCopy(scf::ForOp forOp, Operation *loadOp, Value desc, Value alloc,
                   Value insertIdx, Value extractIdx, Value barrier,
                   Operation *waitOp,
                   llvm::MapVector<Operation *, LoadInfo> &loadToInfo,
                   function_ref<void(OpBuilderWithStage &, int, int, Value,
                                     Value, Value, Value)>
                       createCopy) {
  OpBuilderWithStage builder(forOp);
  auto [stage, clusterId] = tt::getStageCluster(loadOp);
  auto *firstUse = getFirstUseOfPipelinedLoad(loadOp);
  auto [stageForFirstUse, clusterForFirstUse] = tt::getStageCluster(firstUse);

  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  Location loc = loadOp->getLoc();

  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  builder.setInsertionPoint(loadOp);
  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  ttg::MemDescType subviewTy = ttg::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true,
      /*allocShape=*/allocTy.getAllocShape());
  auto view = builder.createWithStage<ttg::MemDescSubviewOp>(
      loc, stage, clusterId, subviewTy, alloc, copyOffsets);

  Value pred = builder.createWithStage<arith::ConstantIntOp>(loc, stage,
                                                             clusterId, 1, 1);
  Value tmaPtr =
      builder.createWithStage<triton::nvidia_gpu::TensorDescToTMAPtrOp>(
          loc, stage, clusterId, desc);
  createCopy(builder, stage, clusterId, tmaPtr, barrier, view, pred);

  auto loadIsMMAv3Shared = loadToInfo[loadOp].isMMAv3Shared;

  builder.setInsertionPointAfter(waitOp);
  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad = builder.createWithStage<ttg::MemDescSubviewOp>(
      loc, stageForFirstUse, clusterForFirstUse, subviewTy, alloc, loadOffsets);
  if (loadIsMMAv3Shared) {
    auto alloc = cast<ttg::LocalAllocOp>((*loadOp->getUsers().begin()));
    tt::replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
    alloc.erase();
  } else {
    SmallVector<ttg::LocalAllocOp> allocsToErase;
    for (Operation *user : loadOp->getUsers()) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        tt::replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
        allocsToErase.push_back(alloc);
      }
    }
    for (auto alloc : allocsToErase) {
      alloc.erase();
    }

    builder.setInsertionPointAfter(viewLoad);
    auto sharedLoad = builder.createWithStage<ttg::LocalLoadOp>(
        loc, stageForFirstUse, clusterForFirstUse,
        loadOp->getResultTypes().front(), viewLoad /*,wait->getResult(0)*/);
    auto result = sharedLoad->getResults();
    loadOp->replaceAllUsesWith(result);
  }
  loadOp->erase();
}

static void
createTMAAsyncLoad(scf::ForOp forOp, tt::ExperimentalDescriptorLoadOp loadOp,
                   Value alloc, Value insertIdx, Value extractIdx,
                   Value barrier, Operation *waitOp,
                   llvm::MapVector<Operation *, LoadInfo> &loadToInfo) {
  return createTMAAsyncCopy(
      forOp, loadOp, loadOp.getDesc(), alloc, insertIdx, extractIdx, barrier,
      waitOp, loadToInfo,
      [&](OpBuilderWithStage &builder, int stage, int clusterId, Value tmaPtr,
          Value barrier, Value view, Value pred) {
        builder.createWithStage<ttng::AsyncTMACopyGlobalToLocalOp>(
            loadOp.getLoc(), stage, clusterId, tmaPtr, loadOp.getIndices(),
            barrier, view, pred);
      });
}

static void createTMAAsyncGather(
    scf::ForOp forOp, tt::ExperimentalDescriptorGatherOp gatherOp, Value alloc,
    Value insertIdx, Value extractIdx, Value barrier, Operation *waitOp,
    llvm::MapVector<Operation *, LoadInfo> &loadToInfo) {
  return createTMAAsyncCopy(
      forOp, gatherOp, gatherOp.getDesc(), alloc, insertIdx, extractIdx,
      barrier, waitOp, loadToInfo,
      [&](OpBuilderWithStage &builder, int stage, int clusterId, Value tmaPtr,
          Value barrier, Value view, Value pred) {
        builder.createWithStage<ttng::AsyncTMAGatherOp>(
            gatherOp.getLoc(), stage, clusterId, tmaPtr, gatherOp.getXOffsets(),
            gatherOp.getYOffset(), barrier, view, pred);
      });
}

static ttg::BlockedEncodingAttr
getBlockedEncoding(tt::LoadOp loadOp, tt::ModuleAxisInfoAnalysis &axisInfo) {
  Value src = loadOp.getPtr();
  auto ty = cast<RankedTensorType>(src.getType());
  auto mod = loadOp->getParentOfType<ModuleOp>();
  int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  tt::AxisInfo::DimVectorT contiguity =
      axisInfo.getAxisInfo(src)->getContiguity();
  SmallVector<unsigned> order = argSort(contiguity);
  unsigned currPerThread = getNumElementsPerThread(loadOp, order, axisInfo);
  SmallVector<unsigned> sizePerThread(order.size(), 1);
  sizePerThread[order[0]] = currPerThread;
  ttg::CTALayoutAttr ctaLayout = ttg::getCTALayout(ty.getEncoding());
  return ttg::BlockedEncodingAttr::get(loadOp->getContext(), ty.getShape(),
                                       sizePerThread, order, numWarps,
                                       threadsPerWarp, ctaLayout);
}

static std::optional<ttg::SharedEncodingTrait>
getSharedEncoding(Operation *loadOp, bool isTMALoad) {
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  auto blockedOrder = ttg::getOrder(ty.getEncoding());
  SmallVector<unsigned> order;
  if (blockedOrder.size() == 3) {
    for (unsigned i = 0; i < blockedOrder.size(); ++i) {
      if (blockedOrder[i] == 0)
        continue;
      order.push_back(blockedOrder[i]);
    }
    order.push_back(0);
  } else {
    order = blockedOrder;
  }

  ttg::SharedEncodingTrait localAllocEnc;
  if (llvm::any_of(loadOp->getUsers(), [&](Operation *user) {
        return isa<ttg::LocalAllocOp>(user);
      })) {
    for (auto user : loadOp->getUsers()) {
      auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
      if (!localAlloc)
        continue;
      auto enc = mlir::cast<ttg::SharedEncodingTrait>(
          localAlloc.getType().getEncoding());
      if (!localAllocEnc) {
        localAllocEnc = enc;
      }
      if (enc != localAllocEnc) {
        // If the allocs don't all have the same encoding, bail.
        return std::nullopt;
      }
    }
  }

  if (isTMALoad) {
    // For TMA, the encoding compatible with it takes precedence over local
    // alloc created for the MMA operand.
    if (localAllocEnc) {
      if (auto sharedMMALayout =
              dyn_cast<ttg::NVMMASharedEncodingAttr>(localAllocEnc)) {
        assert(!sharedMMALayout.getFp4Padded() &&
               "TMA load for mixed precision MMAv5 is not supported yet.");
      }
    }
    return ttg::NVMMASharedEncodingAttr::get(
        ty.getContext(), ty.getShape(), order, ctaLayout, ty.getElementType(),
        /*fp4Padded*/ false);
  }

  // If the load is used by a LocalAllocOp, use the same encoding as the allocs.
  if (localAllocEnc) {
    return localAllocEnc;
  }

  // Use non-swizzled layout for loads that do not feed into dot ops.
  // TODO: This won't be optimal for 2D tensors.
  return ttg::SwizzledSharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order,
                                              ctaLayout);
}

static llvm::SmallVector<Operation *> getDirectUserInBlock(Operation *loadOp) {
  llvm::SmallVector<Operation *> users;
  DenseSet<Operation *> seen;
  for (Operation *user : loadOp->getUsers()) {
    if (!seen.insert(user).second)
      continue;
    if (user->getBlock() == loadOp->getBlock())
      users.push_back(user);
  }
  return users;
}

// When loop doesn't have num_stages attributes, we will look for any load or
// dot (only the first one in the chain). With the attribute we should look for
// any op, but also only the first one.
static llvm::SmallVector<Operation *>
getTransitiveUserInBlock(Operation *baseOp, scf::ForOp &forOp) {
  llvm::SmallVector<Operation *> users;
  DenseSet<Operation *> seen;
  bool loopHasAttribute = forOp->hasAttr(tt::kNumStagesAttrName);
  std::function<void(Operation *, Operation *, bool)> dfs =
      [&](Operation *op, Operation *baseOp, bool anyOp) {
        if (!seen.insert(op).second)
          return;
        if (op != baseOp) {
          if (anyOp) {
            // Only track the first op in the dependence chain.
            users.push_back(op);
            return;
          }
          if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp,
                  tt::ExperimentalDescriptorGatherOp>(op) ||
              isa<mlir::triton::DotOpInterface>(op)) {
            // Stop recursion when hitting a LoadOp or a DotOp.
            users.push_back(op);
            return;
          }
        }
        for (Operation *user : op->getUsers())
          if (user->getBlock() == op->getBlock())
            dfs(user, baseOp, anyOp);
        if (auto tmemCopy = dyn_cast<triton::nvidia_gpu::TMEMCopyOp>(op)) {
          auto tmemAlloc =
              tmemCopy.getDst()
                  .getDefiningOp<triton::nvidia_gpu::TMEMAllocOp>();
          dfs(tmemAlloc, baseOp, anyOp);
        }
      };
  // We are matching the behavior before refactoring:
  //   For loops without num_stage attributes, we check for dot users.
  //   For loops with num_stage attributes, we check for dot users, if there are
  //   no dot users, we check for direct users.
  dfs(baseOp, baseOp, false /*anyOp*/);
  if (loopHasAttribute) {
    seen.clear();
    dfs(baseOp, baseOp, true /*anyOp*/);
  }
  return users;
}

static bool isMMAv3Buffer(Operation *loadOp) {
  if (!loadOp->hasOneUse())
    return false;
  Operation *user = *loadOp->getUsers().begin();
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
    return isa<ttg::NVMMASharedEncodingAttr>(alloc.getType().getEncoding());
  }
  return false;
}

static llvm::MapVector<Operation *, LoadInfo>
assignMemoryLayouts(scf::ForOp &forOp,
                    tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  llvm::MapVector<Operation *, LoadInfo> loadToInfo;

  // Go through all loads in the loop, check to see if they are pipelined.
  llvm::DenseSet<Operation *> loadsToPipeline;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (!isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp,
             tt::ExperimentalDescriptorGatherOp>(op))
      continue;
    if (loadToInfo.count(&op))
      // TODO pawel: err, we'd need to verify that the distance is the same
      continue;
    if (!op.hasAttr(mlir::triton::kLoopStageAttrName))
      continue;

    // Check stage for uses. If the first use is in a different stage, treat it
    // as a pipelined load.
    auto [sLoad, _cLoad] = tt::getStageCluster(&op);
    Operation *firstUse = getFirstUseOfPipelinedLoad(&op);
    LDBG("first use for load " << op);
    LDBG("  - use: " << *firstUse);
    auto firstUseStageCluster = tt::maybeGetStageCluster(firstUse);
    if (!firstUseStageCluster || firstUseStageCluster->first == sLoad)
      continue;

    // Try to set shared encoding etc for the pipelined load.
    auto users = getTransitiveUserInBlock(&op, forOp);
    LLVM_DEBUG({
      LDBG("TransitiveUser for load " << op);
      for (const auto user : users) {
        LDBG("  - use: " << *user);
      }
    });

    bool isTMALoad = isa<tt::ExperimentalDescriptorLoadOp,
                         tt::ExperimentalDescriptorGatherOp>(op);
    // TODO: b/381421713 - Uncomment this once pipelining is fixed.
    // loadsToPipeline.insert(&op);
    LoadInfo loadInfo;
    for (auto use : users) {
      if (isa<mlir::triton::DotOpInterface>(use)) {
        LDBG("set shared encoding with dot user: " << *use);
        auto dot = dyn_cast<tt::DotOp>(use);
        bool mmav3Shmem = isMMAv3Buffer(&op);
        loadInfo.usedByDot = true;
        loadInfo.isMMAv3Shared = mmav3Shmem;

        if (mmav3Shmem || isTMALoad) {
          loadInfo.sharedEncoding =
              getSharedEncoding(&op, isTMALoad).value_or(nullptr);
        } else if (!mmav3Shmem || dot) {
          bool incompatible = false;

          loadInfo.sharedEncoding =
              getSharedEncIfAllUsersAreDotEnc(op.getResult(0), incompatible)
                  .value_or(nullptr);
        }
      }

      // If we still don't have a shared encoding, try a "generic" shared
      // encoding.
      if (!loadInfo.sharedEncoding) {
        assert(!loadInfo.isMMAv3Shared &&
               "For MMAv3 pipelining we should have shared encoding");
        LDBG("try generic shared encoding");
        loadInfo.sharedEncoding =
            getSharedEncoding(&op, isTMALoad).value_or(nullptr);
        if (isa<ttng::TCGen5MMAScaledOp>(use)) {
          loadInfo.isMMAv5Scale = true;
        }
        if (auto loadOp = dyn_cast<tt::LoadOp>(op))
          loadInfo.blockedEncoding =
              getBlockedEncoding(loadOp, axisInfoAnalysis);
      }
    }

    // TODO: b/381421713 - Remove this once pipelining is fixed.
    if (!loadInfo.sharedEncoding) continue;
    loadsToPipeline.insert(&op);

    loadToInfo[&op] = loadInfo;
  }
  // Make sure all loads in loadsToPipeline are in loadToInfo.
  for (auto *load : loadsToPipeline)
    assert(loadToInfo.count(load) &&
           "pipelined loads should have sharedEncoding");

  return loadToInfo;
}

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, Operation *loadOp,
                         ttg::SharedEncodingTrait sharedEnc,
                         unsigned distance) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type memdescType = ttg::MemDescType::get(bufferShape, ty.getElementType(),
                                           sharedEnc, sharedMemorySpace,
                                           /*mutableMemory=*/true);
  Value alloc =
      builder.create<ttg::LocalAllocOp>(loadOp->getLoc(), memdescType);
  return alloc;
}

// Create an allocation to hold the mbarriers.
static Value createBarrierAlloc(scf::ForOp &forOp, unsigned distance) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  Location loc = forOp.getLoc();
  auto context = forOp.getContext();
  auto barrierCTALayout =
      ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                              /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
      context, 1, 1, 1, {0}, barrierCTALayout);
  auto barrierMemDescType = ttg::MemDescType::get(
      {distance}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  Type singleBarrierMemDescType = ttg::MemDescType::get(
      {1}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true,
      /*allocShape=*/barrierMemDescType.getAllocShape());
  Value barrierAlloc =
      builder.create<ttg::LocalAllocOp>(loc, barrierMemDescType);
  for (unsigned i = 0; i < distance; i++) {
    Value idx = builder.create<arith::ConstantIntOp>(loc, i, 32);
    Value barrierView = builder.create<ttg::MemDescSubviewOp>(
        loc, singleBarrierMemDescType, barrierAlloc, idx);
    builder.create<ttng::InitBarrierOp>(forOp->getLoc(), barrierView, 1);
  }
  return barrierAlloc;
}

struct StageGroup {
  Value insertIdx;
  Value extractIdx;
  Value phase;
  bool hasTMALoad = false;
};
struct AsyncLoad {
  Operation *loadOp;
  Value alloc;
  Value barrier;
  Operation *waitOp = nullptr;
  int firstUseStage, firstUseCluster;
  bool isTMALoad = false;
  int numBuffers = 0;
};

// Create barriers and wait ops for the async loads. Barriers may be shared by
// multiple loads if the schedule allows it.
static void createTMABarrierAndWait(
    scf::ForOp &forOp, SmallVector<AsyncLoad> &asyncLoads,
    SmallVector<Value> &barriers,
    const llvm::MapVector<int, StageGroup> &stageGroups,
    const llvm::MapVector<Operation *, LoadInfo> &loadToInfo) {
  llvm::SmallDenseMap<Operation *, AsyncLoad *> loadToAsyncLoad;
  for (AsyncLoad &asyncLoad : asyncLoads) {
    loadToAsyncLoad[asyncLoad.loadOp] = &asyncLoad;
  }
  SmallVector<SmallVector<AsyncLoad *>> loadGroups;
  llvm::SmallDenseSet<Operation *> visited;
  // Find groups of loads that can share the same barrier. We look consecutive
  // loads and check that there are uses in between.
  for (AsyncLoad &asyncLoad : asyncLoads) {
    if (!asyncLoad.isTMALoad || visited.count(asyncLoad.loadOp))
      continue;
    llvm::SmallDenseSet<Operation *> users;
    SmallVector<AsyncLoad *> group;
    Block *loadBlock = asyncLoad.loadOp->getBlock();
    auto addToGroup = [&](AsyncLoad *loadInfo) {
      group.push_back(loadInfo);
      visited.insert(loadInfo->loadOp);
      for (Operation *user : loadInfo->loadOp->getUsers()) {
        auto it = loadToInfo.find(loadInfo->loadOp);
        if (it != loadToInfo.end()) {
          // Special case for MMAv3 loads, we can ignore the alloc and only
          // consider uses of the alloc op since it will be removed.
          if (it->second.isMMAv3Shared) {
            auto alloc = cast<ttg::LocalAllocOp>(
                (*loadInfo->loadOp->getUsers().begin()));
            if (alloc->getBlock() == loadBlock) {
              users.insert(alloc->getUsers().begin(), alloc->getUsers().end());
              continue;
            }
          }
        }
        Operation *userInBlock = loadBlock->findAncestorOpInBlock(*user);
        if (userInBlock)
          users.insert(userInBlock);
      }
    };
    addToGroup(&asyncLoad);
    Operation *nextOp = asyncLoad.loadOp->getNextNode();
    int numBuffers = asyncLoad.numBuffers;
    while (nextOp) {
      if (users.count(nextOp) || visited.count(nextOp))
        break;
      if (isa<tt::ExperimentalDescriptorLoadOp,
              tt::ExperimentalDescriptorGatherOp>(nextOp)) {
        auto it = loadToAsyncLoad.find(nextOp);
        if (it != loadToAsyncLoad.end() && it->second->isTMALoad) {
          if (it->second->numBuffers != numBuffers)
            break;
          if (group.size() > 0 &&
              sameStageCluster(group[0]->loadOp, it->second->loadOp))
            addToGroup(it->second);
        }
      }
      nextOp = nextOp->getNextNode();
    }
    loadGroups.push_back(group);
  }

  // For each group calculate the size and insert the barrier after the last
  // load.
  for (SmallVector<AsyncLoad *> &group : loadGroups) {
    int sizeInBytes = 0;
    int numBuffers = group[0]->numBuffers;
    const StageGroup &stageGroup = stageGroups.find(numBuffers)->second;
    for (AsyncLoad *asyncLoad : group) {
      auto tensorTy =
          cast<RankedTensorType>(asyncLoad->loadOp->getResult(0).getType());
      int loadSize = product(tensorTy.getShape());
      sizeInBytes +=
          loadSize * tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
    }

    auto [stage, cluster] = tt::getStageCluster(group[0]->loadOp);
    Value barrierAlloc = createBarrierAlloc(forOp, numBuffers);
    barriers.push_back(barrierAlloc);
    Location loc = forOp.getLoc();
    OpBuilderWithStage builder(forOp);
    Attribute sharedMemorySpace =
        ttg::SharedMemorySpaceAttr::get(builder.getContext());
    auto allocTy = cast<ttg::MemDescType>(barrierAlloc.getType());
    ttg::MemDescType barrierTy = ttg::MemDescType::get(
        {1}, builder.getI64Type(), allocTy.getEncoding(), sharedMemorySpace,
        /*mutableMemory=*/true,
        /*allocShape=*/allocTy.getAllocShape());
    builder.setInsertionPoint(group[0]->loadOp);
    Value barrier = builder.createWithStage<ttg::MemDescSubviewOp>(
        loc, stage, cluster, barrierTy, barrierAlloc,
        ArrayRef<Value>({stageGroup.insertIdx}));
    Value pred = builder.createWithStage<arith::ConstantIntOp>(loc, stage,
                                                               cluster, 1, 1);
    Operation *expect = builder.createWithStage<ttng::BarrierExpectOp>(
        forOp.getLoc(), stage, cluster, barrier, sizeInBytes, pred);

    builder.setInsertionPointAfter(group.back()->loadOp);
    Value barrierViewWait = builder.createWithStage<ttg::MemDescSubviewOp>(
        loc, group[0]->firstUseStage, group[0]->firstUseCluster, barrierTy,
        barrierAlloc, ArrayRef<Value>({stageGroup.extractIdx}));
    Operation *wait = builder.createWithStage<ttng::WaitBarrierOp>(
        loc, group[0]->firstUseStage, group[0]->firstUseCluster,
        barrierViewWait, stageGroup.phase);
    // Update the async loads info.
    for (AsyncLoad *asyncLoad : group) {
      asyncLoad->barrier = barrier;
      asyncLoad->waitOp = wait;
    }
  }
}

// This is similar to CoarseSchedule.createFinalSchedule.
static std::vector<std::pair<Operation *, unsigned>>
getFinalSchedule(scf::ForOp &forOp, int numStages) {
  auto [minClusterId, maxClusterId] = tt::getMinMaxCluster(forOp);
  SmallVector<SmallVector<Operation *>, 8> orderClusters(maxClusterId -
                                                         minClusterId + 1);
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (!op.hasAttr(mlir::triton::kLoopStageAttrName) ||
        !op.hasAttr(mlir::triton::kLoopClusterAttrName))
      continue;

    auto [stage, clusterId] = tt::getStageCluster(&op);
    assert(stage < numStages && "Op with invalid stage!");
    orderClusters[clusterId - minClusterId].push_back(&op);
  }
  std::vector<std::pair<Operation *, unsigned>> fSchedule;
  for (int i = 0; i < orderClusters.size(); i++) {
    for (auto op : orderClusters[i]) {
      auto [stage, _] = tt::getStageCluster(op);
      fSchedule.push_back({op, stage});
    }
  }
  return fSchedule;
}

LogicalResult
allocTMABuffers(scf::ForOp forOp,
                llvm::MapVector<Operation *, Value> &tmaBufferMapping,
                int numStages) {
  IRRewriter rewriter(forOp);

  // Create a multi-buffered allocation for each MakeTensorDescOp call in the
  // loop
  forOp.walk([&](tt::MakeTensorDescOp op) {
    // TODO peter: walk to loop yield to find the init value if this is a
    // loop-carried value. That would save us from allocating another buffer
    // just for the init value
    auto loc = op.getLoc();
    Value alloc = rewriter.create<triton::gpu::GlobalScratchAllocOp>(
        loc, triton::getPointerType(rewriter.getI8Type()),
        numStages * ttng::TMA_SIZE_BYTES, ttng::TMA_ALIGN);
    tmaBufferMapping[op.getOperation()] = alloc;
  });
  return success();
}

template <typename BuilderT>
Value createIncrementModulo(BuilderT &builder, Location loc, Value counter,
                            Value modulus, Value zero, Value one) {
  Value addOne = builder.template create<arith::AddIOp>(loc, counter, one);
  Value inRangeCond = builder.template create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, addOne, modulus);
  return builder.template create<arith::SelectOp>(loc, inRangeCond, addOne,
                                                  zero);
}

template <typename BuilderT>
Value subviewTMADescriptor(BuilderT &builder, Location loc, Value alloc,
                           Value counter) {
  Value tmaSizeVal = builder.template create<arith::ConstantIntOp>(
      loc, ttng::TMA_SIZE_BYTES, 32);
  Value offset =
      builder.template create<arith::MulIOp>(loc, tmaSizeVal, counter);
  return builder.template create<triton::AddPtrOp>(loc, alloc.getType(), alloc,
                                                   offset);
}

LogicalResult rewriteTMABufferUpdates(
    scf::ForOp forOp,
    const llvm::MapVector<Operation *, Value> &tmaBufferMapping,
    ArrayRef<BlockArgument> tmaCounters, int numStages, Value one, Value zero) {
  assert(tmaBufferMapping.size() == tmaCounters.size());

  Value numStagesVal = mlir::OpBuilder(forOp).create<arith::ConstantIntOp>(
      forOp.getLoc(), numStages, 32);

  for (auto [iOp, pair] : llvm::enumerate(tmaBufferMapping)) {
    auto &[op, alloc] = pair;

    // Rewriter MakeTensorDescOp as writing a TMA descriptor
    auto makeDescOp = cast<tt::MakeTensorDescOp>(op);

    OpBuilderForStage stageBuilder(makeDescOp);
    auto loc = makeDescOp.getLoc();

    BlockArgument counter = tmaCounters[iOp];
    Value nextBuf = subviewTMADescriptor(stageBuilder, loc, alloc, counter);
    if (failed(ttng::createTMADesc(nextBuf, makeDescOp, stageBuilder))) {
      return failure();
    }
    stageBuilder.create<triton::ExperimentalTensormapFenceproxyAcquireOp>(
        loc, nextBuf);
    Value nextDesc = stageBuilder.create<triton::ReinterpretTensorDescOp>(
        loc, makeDescOp.getType(), nextBuf);

    makeDescOp.getResult().replaceAllUsesWith(nextDesc);

    // Increment the buffer index counter
    Value nextCounter = createIncrementModulo(stageBuilder, loc, counter,
                                              numStagesVal, zero, one);

    // If we are in a (potentially nested) if region, propagate the counter
    // up to the main for op body scope
    Operation *curOp = op;
    Operation *parent = op->getParentOp();
    while (parent != forOp.getOperation()) {
      auto ifOp = dyn_cast<scf::IfOp>(parent);
      if (!ifOp) {
        std::string msg;
        llvm::raw_string_ostream ss(msg);
        ss << "Cannot pipeline MakeTensorDescOp inside:\n";
        parent->print(ss);
        ss << "\nOnly scf.if regions are supported";
        return makeDescOp->emitOpError(std::move(msg));
      }

      IRRewriter rewriter(parent);
      auto newIfOp =
          replaceIfOpWithNewSignature(rewriter, ifOp, {nextCounter.getType()});

      auto yieldNewBlock = newIfOp.thenBlock();
      auto yieldOldBlock = newIfOp.elseBlock();

      if (yieldNewBlock != curOp->getBlock()) {
        std::swap(yieldNewBlock, yieldOldBlock);
      }
      cast<scf::YieldOp>(yieldNewBlock->getTerminator())
          .getResultsMutable()
          .append(nextCounter);
      cast<scf::YieldOp>(yieldOldBlock->getTerminator())
          .getResultsMutable()
          .append(counter);

      ifOp.erase();
      nextCounter = newIfOp.getResults().back();
      curOp = newIfOp;
      parent = newIfOp->getParentOp();
    }

    // Finally, rewrite the loop level yield
    auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    forYield.setOperand(counter.getArgNumber() - 1, nextCounter);
  }
  return success();
}

// Convert load ops into their async version and apply multi-buffering based on
// the required number of buffers.
static SmallVector<Value>
createAsyncOps(scf::ForOp &forOp,
               llvm::MapVector<Operation *, LoadInfo> &loadToInfo,
               SmallVector<Value> &barriers, int numStages) {
  llvm::MapVector<Operation *, Value> tmaBufferMapping;
  if (failed(allocTMABuffers(forOp, tmaBufferMapping, numStages))) {
    llvm_unreachable("TMA pipelining failed");
  }

  // Each group of loads/allocs with the same number of buffers (and stages)
  // will share the indices and barriers.

  SmallVector<AsyncLoad> asyncLoads;
  SmallVector<Value> allocs;
  llvm::MapVector<int, StageGroup> stageGroups;

  for (auto &[loadOp, info] : loadToInfo) {
    AsyncLoad asyncLoad;
    asyncLoad.loadOp = loadOp;
    bool isTMALoad = false;
    int numBuffers = info.distToUse;
    // For MMAv3, we need an extra buffer as this is assumed in the wgmma
    // pipelining post-processing. Additionally, SMEM for scales in MMAv5
    // should get the same number of buffers as the operand SMEM.
    if (info.isMMAv3Shared || info.isMMAv5Scale) {
      ++numBuffers;
    }
    if (isa<tt::ExperimentalDescriptorLoadOp,
            tt::ExperimentalDescriptorGatherOp>(loadOp)) {
      isTMALoad = true;
      asyncLoad.isTMALoad = isTMALoad;
    }
    assert(info.sharedEncoding && "LoadOp shared encoding not defined.");
    Value alloc = createAlloc(forOp, loadOp, info.sharedEncoding, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    allocs.push_back(alloc);
    asyncLoad.alloc = alloc;

    auto *firstUse = getFirstUseOfPipelinedLoad(loadOp);
    auto [firstUseStage, firstUseCluster] = tt::getStageCluster(firstUse);
    asyncLoad.firstUseStage = firstUseStage;
    asyncLoad.firstUseCluster = firstUseCluster;
    asyncLoad.numBuffers = numBuffers;
    stageGroups.insert({numBuffers, {}});
    if (isTMALoad) {
      stageGroups[numBuffers].hasTMALoad = true;
    }
    asyncLoads.push_back(asyncLoad);
  }

  IRRewriter builder(forOp.getContext());
  builder.setInsertionPoint(forOp);

  Location loc = forOp.getLoc();
  // Create a counter to index into the allocations per loop iteration.
  // NOTE: We create two duplicates values, insertIdx and extractIdx so that the
  // pipeliner will re-materialize the value in later stages of the pipeline
  // instead of carrying it as a dependency across multiple iterations.
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  SmallVector<Value> newOperands;
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  for (auto [_, stageGroup] : stageGroups) {
    newOperands.push_back(minusOne); // insertIdx
    newOperands.push_back(minusOne); // extractIdx
    if (stageGroup.hasTMALoad) {
      // A single barrier arrival sequence is a "phase" and two phases can
      // overlap, provided the phases are differentiated with an alternating
      // boolean value.
      newOperands.push_back(zero); // phase
    }
  }
  // Also create one counter per TMA buffer. This allows the descriptors to be
  // updated independently without needing to write duplicate of existing tma
  // descriptors.
  unsigned tmaCounterArgsStartIdx = newOperandIndex + newOperands.size();
  for (int i = 0; i < tmaBufferMapping.size(); ++i) {
    newOperands.push_back(zero);
  }

  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;

  auto tmaCounters = ArrayRef<BlockArgument>(newForOp.getBody()->getArguments())
                         .slice(tmaCounterArgsStartIdx);

  // Update yield op with temporary yield values
  auto forYield = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  for (unsigned i = 0; i < newOperands.size(); ++i) {
    forYield.getResultsMutable().append(newOperands[i]);
  }

  if (failed(rewriteTMABufferUpdates(newForOp, tmaBufferMapping, tmaCounters,
                                     numStages, one, zero))) {
    llvm_unreachable("Failed to rewrite TMA ops");
  }
  tmaBufferMapping.clear();

  builder.setInsertionPoint(forOp);
  loc = forOp.getLoc();
  int argIdx = newOperandIndex;
  for (auto &[numBuffers, stageGroup] : stageGroups) {
    Value insertIdx = newForOp.getBody()->getArgument(argIdx);
    argIdx++;
    Value extractIdx = newForOp.getBody()->getArgument(argIdx);
    argIdx++;
    Value phase = nullptr;
    if (stageGroup.hasTMALoad) {
      phase = newForOp.getBody()->getArgument(argIdx);
      argIdx++;
    }

    // Create two counters for the insert and extract indices to avoid creating
    // long liverange.
    builder.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());

    Value numBuffersVal =
        builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
    insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
    Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 insertIdx, numBuffersVal);
    insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);
    stageGroup.insertIdx = insertIdx;

    extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
    // Duplicate the constant to keep it from being carried across loops.
    numBuffersVal = builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
    Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 extractIdx, numBuffersVal);
    extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);
    stageGroup.extractIdx = extractIdx;
    if (phase) {
      Value nextPhase = builder.create<arith::XOrIOp>(loc, phase, one);
      phase = builder.create<arith::SelectOp>(loc, cndExt, phase, nextPhase);
      stageGroup.phase = phase;
    }
  }
  createTMABarrierAndWait(forOp, asyncLoads, barriers, stageGroups, loadToInfo);

  auto [_, maxClusterId] = tt::getMinMaxCluster(forOp);
  for (AsyncLoad &asyncLoad : asyncLoads) {
    auto [insertIdx, extractIdx, phase, _] = stageGroups[asyncLoad.numBuffers];
    if (auto loadOp = dyn_cast<tt::LoadOp>(asyncLoad.loadOp)) {
      createAsyncCopy(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                      loadToInfo, maxClusterId);
    } else if (auto descLoad = dyn_cast<tt::ExperimentalDescriptorLoadOp>(
                   asyncLoad.loadOp)) {
      createTMAAsyncLoad(forOp, descLoad, asyncLoad.alloc, insertIdx,
                         extractIdx, asyncLoad.barrier, asyncLoad.waitOp,
                         loadToInfo);
    } else {
      auto descGather =
          cast<tt::ExperimentalDescriptorGatherOp>(asyncLoad.loadOp);
      createTMAAsyncGather(forOp, descGather, asyncLoad.alloc, insertIdx,
                           extractIdx, asyncLoad.barrier, asyncLoad.waitOp,
                           loadToInfo);
    }
  }
  // Patch the yield with the updated counters. Subtract to account for the loop
  // counter.
  argIdx = newOperandIndex - 1;
  for (auto &[numBuffers, stageGroup] : stageGroups) {
    forYield.setOperand(argIdx++, stageGroup.insertIdx);
    forYield.setOperand(argIdx++, stageGroup.extractIdx);
    if (stageGroup.phase)
      forYield.setOperand(argIdx++, stageGroup.phase);
  }
  assert(argIdx + 1 == tmaCounterArgsStartIdx);

  tt::CoarseSchedule coarseSchedule(numStages);
  coarseSchedule.deSerialize(forOp);
  scheduleDependencies(forOp, coarseSchedule);
  coarseSchedule.serialize(forOp);

  // Make sure all ops have attributes.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    assert(op.hasAttr(mlir::triton::kLoopStageAttrName) &&
           op.hasAttr(mlir::triton::kLoopClusterAttrName));
  }
  return allocs;
}

static void invalidateBarriers(OpBuilder &builder,
                               SmallVector<Value> &barriers) {
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(builder.getContext());
  for (Value barrier : barriers) {
    auto allocTy = cast<ttg::MemDescType>(barrier.getType());
    int numBarriers = allocTy.getShape()[0];
    for (int i = 0; i < numBarriers; i++) {
      Value idx = builder.create<arith::ConstantIntOp>(barrier.getLoc(), i, 32);
      ttg::MemDescType barrierTy = ttg::MemDescType::get(
          {1}, builder.getI64Type(), allocTy.getEncoding(), sharedMemorySpace,
          /*mutableMemory=*/true,
          /*allocShape=*/allocTy.getShape());
      Value barrierView = builder.create<ttg::MemDescSubviewOp>(
          barrier.getLoc(), barrierTy, barrier, idx);
      builder.create<ttng::InvalBarrierOp>(barrier.getLoc(), barrierView);
    }
  }
}

bool mlir::triton::preProcessLoopAndGetSchedule(
    scf::ForOp &forOp, int numStages, mlir::triton::PipeliningOption &options) {

  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  // Check which loads are good for pipelining, and assign them
  // memory layouts.
  llvm::MapVector<Operation *, LoadInfo> loadToInfo =
      assignMemoryLayouts(forOp, axisInfoAnalysis);
  if (loadToInfo.empty())
    return false;

  // Distance from the load to the use.
  for (auto &[loadOp, info] : loadToInfo) {
    auto *use = getFirstUseOfPipelinedLoad(loadOp);
    auto [stage, _] = tt::getStageCluster(loadOp);
    auto [stageUse, t_] = tt::getStageCluster(use);
    loadToInfo[loadOp].distToUse = stageUse - stage;
  }

  SmallVector<Value> barriers;
  // Convert the loads into async loads and create the allocs.
  SmallVector<Value> allocs =
      createAsyncOps(forOp, loadToInfo, barriers, numStages);
  LDBG("after lowering: " << forOp->getParentOfType<ModuleOp>());

  // Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      getFinalSchedule(forOp, numStages);

  // Fill out the pipeline options.
  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = tt::predicateOp;
  options.supportDynamicLoops = true;
  options.annotateFn = [](Operation *op,
                          mlir::triton::PipeliningOption::PipelinerPart part,
                          unsigned iteration) {};

  // Clean up the attributes.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    op.removeAttr(mlir::triton::kLoopStageAttrName);
    op.removeAttr(mlir::triton::kLoopClusterAttrName);
  }

  // Insert a wait 0 after the loop
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), ValueRange({}), 0);
  // Invalidate any mbarrier create
  invalidateBarriers(builder, barriers);
  // Explicitly deallocate allocated tensors after the wait op
  for (auto alloc : allocs)
    builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);
  return true;
}

/// Find the minimum number of async_commit_group ops between the wait
/// and the associated async_commit_group. This can be safely used as the wait
/// number.
static int minNumInterleavedCommitOps(Operation *waitOp) {
  auto countCommitsBetween = [](Operation *op1, Operation *op2) {
    int count = 0;
    for (auto op = op1; op != op2; op = op->getNextNode()) {
      if (isa<ttg::AsyncCommitGroupOp>(op))
        count++;
      // Intentionally skip block ops' children. This will give us
      // convervatively low number of insert ops.
    }
    return count;
  };

  int minCommitNumber = INT_MAX;

  // DFS the def chain of the extract op to find the insert op. On each path
  // we calculate the number of async_commit. Then we select the minimum number
  // of async_commit ops among all the paths.
  std::function<int(Value, Operation *, int)> minOverHistories =
      [&](Value val, Operation *sinkOp, int thisHistorySum) -> int {
    if (Operation *defOp = val.getDefiningOp()) {
      thisHistorySum += countCommitsBetween(defOp->getNextNode(), sinkOp);
      minCommitNumber = std::min(minCommitNumber, thisHistorySum);
      return minCommitNumber;
    }
    if (auto arg = mlir::dyn_cast<BlockArgument>(val)) {
      Block *block = arg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

      // Failed to track, return 0 conservatively.
      if (!forOp)
        return 0;

      Operation *firstForInst = &*forOp.getBody()->begin();
      int insertsBetween = countCommitsBetween(firstForInst, sinkOp);
      thisHistorySum += insertsBetween;
      if (thisHistorySum >= minCommitNumber)
        return minCommitNumber;

      // get the value assigned to the argument coming from outside the loop
      Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
      int min1 = minOverHistories(incomingVal, forOp, thisHistorySum);

      // get the value assigned to the argument coming from the previous
      // iteration
      Operation *yieldOp = block->getTerminator();
      Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
      int min2 = minOverHistories(prevVal, yieldOp, thisHistorySum);
      return std::min(std::min(min1, min2), minCommitNumber);
    }
    // Failed to track, return 0 conservatively.
    return 0;
  };

  if (waitOp->getNumOperands() != 1)
    return 0;
  int minCommits = minOverHistories(waitOp->getOperand(0), waitOp, 0);
  return minCommits;
}

// Look for consecutive wait ops and combine them into a single wait op.
static void
combineRedundantWaitOps(llvm::SmallSetVector<ttg::AsyncWaitOp, 8> &waitOps) {
  llvm::MapVector<ttg::AsyncWaitOp, ttg::AsyncWaitOp> toDelete;
  for (auto waitOp : waitOps) {
    if (toDelete.count(waitOp))
      continue;
    SmallVector<ttg::AsyncWaitOp> waitGroup = {waitOp};
    SmallVector<Value> depTokens;
    unsigned minWaitNumber = waitOp.getNum();
    Operation *next = waitOp->getNextNode();
    while (next && isa<ttg::MemDescSubviewOp, ttg::AsyncWaitOp>(next)) {
      if (auto nextWait = dyn_cast<ttg::AsyncWaitOp>(next)) {
        waitGroup.push_back(nextWait);
        minWaitNumber = std::min(minWaitNumber, nextWait.getNum());
        depTokens.append(nextWait.getOperands().begin(),
                         nextWait.getOperands().end());
      }
      next = next->getNextNode();
    }
    if (waitGroup.size() == 1)
      continue;
    OpBuilder builder(waitGroup.back());
    auto newWaitOp = builder.create<ttg::AsyncWaitOp>(waitOp.getLoc(),
                                                      depTokens, minWaitNumber);
    for (auto waitOp : waitGroup) {
      toDelete[waitOp] = newWaitOp;
    }
  }
  for (auto waitOp : toDelete) {
    waitOp.first->replaceAllUsesWith(waitOp.second);
    waitOp.first->erase();
  }
}

/// Update wait op number by analyzing the number of async_commit_group ops
/// along all paths.
void mlir::triton::updateWaits(ModuleOp module) {
  llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
  module.walk([&](ttg::AsyncWaitOp waitOp) {
    int minNumCommits = minNumInterleavedCommitOps(waitOp);
    waitOp.setNum(minNumCommits);
    waitOps.insert(waitOp);
  });
  combineRedundantWaitOps(waitOps);
}

// Add the given values as operands of the given wait, and replace all uses of
// the values with the wait.  Also adds related MemDesc's to the wait.
//
// Threading %a through the wait transforms
//
//   %a = <...>
//   (%x', %y') = ttng.async_wait %x, %y
//   %b = fn(%a)
//
// into
//
//   %a = <...>
//   (%x', %y', %a') = ttng.async_wait %x, %y, %a
//   %b = fn(%a')
//
// The wait must dominate all uses of the elements of `values`.
//
// In addition to adding each value from `values` to the wait, this function
// also adds some MemDesc's to the wait.  The idea is that if you have
//
//   %alloc = ttg.local_alloc ...
//   %a = ttng.warp_group_dot %alloc
//   %a1 = ttng.warp_group_dot_wait %a
//
// then we want the wait to depend on %alloc as well as %a.  This extends the
// live range of %alloc, so that it won't be destroyed until after the dot is
// waited on.
//
// Specifically, this function finds all warp_group_dot ops that elements of
// `values` depend on.  Then it adds the MemDesc operands of those dots to the
// wait.
static void threadValuesThroughWait(ttng::WarpGroupDotWaitOp wait,
                                    MutableArrayRef<Value> values) {
  IRRewriter builder(wait.getContext());
  builder.setInsertionPoint(wait);

  // Operands are only added to the wait through this function, so we can have
  // the invariant that the wait has no duplicates.  This makes things a bit
  // easier below.
  size_t origNumOperands = wait.getNumOperands();
  SetVector<Value> newOperands(wait.getOperands().begin(),
                               wait.getOperands().end());
  assert(newOperands.size() == origNumOperands &&
         "Wait op has duplicate operands.");

  newOperands.insert(values.begin(), values.end());

  // Find memdefs depended on by `values` through async dot ops.
  SmallVector<ttng::WarpGroupDotOp> asyncDots;
  for (Value v : values) {
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *op) {
      if (auto dot = dyn_cast<ttng::WarpGroupDotOp>(op)) {
        asyncDots.push_back(dot);
        return false;
      }
      return op->getBlock() == wait->getBlock();
    };
    SetVector<Operation *> slice;
    getBackwardSlice(v, &slice, options);
  }

  for (ttng::WarpGroupDotOp dot : asyncDots) {
    for (Value operand : dot.getOperands()) {
      if (isa<ttg::MemDescType>(operand.getType())) {
        newOperands.insert(operand);
      }
    }
  }

  // We can't use replaceWithNewOp because we're changing the number of return
  // values in the operation.
  auto newWait = builder.create<ttng::WarpGroupDotWaitOp>(
      wait.getLoc(), llvm::to_vector(newOperands), wait.getPendings());

  auto dominatedByNewWait = [&](OpOperand &operand) {
    auto opInThisBlock =
        newWait->getBlock()->findAncestorOpInBlock(*operand.getOwner());
    return opInThisBlock && newWait->isBeforeInBlock(opInThisBlock);
  };
  for (int i = 0; i < origNumOperands; i++) {
    Value operand = wait.getResult(i);
    if (!isa<ttg::MemDescType>(operand.getType()))
      operand.replaceAllUsesWith(newWait.getResult(i));
  }
  for (int i = origNumOperands; i < newOperands.size(); i++) {
    Value operand = newWait.getOperand(i);
    if (!isa<ttg::MemDescType>(operand.getType()))
      operand.replaceUsesWithIf(newWait.getResult(i), dominatedByNewWait);
  }
  wait->erase();
}

// Determines whether a given MMAv3 dot op, represented as ttng.warp_group_dot,
// needs a wait immediately after it.
//
// In PTX, MMAv3 exists only as an asynchronous op.  In Triton, we can represent
// MMAv3 ops as either ttng.warp_group_dot {isAsync=True} or ttng.warp_group_dot
// {isAsync=False}.  But even if we use ttng.warp_group_dot {isAsync=True}, the
// conservative thing is to make a dot "effectively synchronous" by inserting a
// `ttng.warp_group_dot_wait {pendings=0}` right after it.
//
// We can omit the wait and create a "properly async" dot if all of the
// following are true.
//
//  1. All operands that touch shared memory are multi-buffered, i.e. can't read
//     an incomplete value while it's being written asynchronously by a load.
//     1a. If operand A is in registers, these registers cannot be updated
//     inside
//         the loop.
//         **Exception** if the operand is produced by a preceding WGMMA,
//         then this op can be properly async. Either the f16 shortcut is
//         possible and the WGMMA's can run back-to-back (see rule 3 below), or
//         elementwise truncate is needed, in which case the preceding WGMMA is
//         not async and a WarpGroupDotWait is inserted right after, which
//         guarantees exclusive access to the operand registers.
//
//  2. If the dot is used by any op in the loop, it must be used under an `if`,
//     and will be synced with a `wait 0` at the beginning of the `if` block.
//
//  3. During iteration i, between the start of the loop up until the first
//     `ttng.warp_group_dot_wait {pendings=0}` op, the result of the dot from
//     iteration i-1 is consumed only by other MMAv3 dots as the `c` operand.
//
//     This is safe because the following pseudo-PTX is valid:
//
//        %accum = warp_group_dot %a1, %b1, %c1
//        %accum = warp_group_dot %a2, %b2, %accum
//
//     That is, the second async dot can use the result of the first one without
//     an intervening wait.  However, the only operation that can legally read
//     %accum before the wait is another warp_group_dot, and this only works for
//     the `c` operand, not `a` or `b`.  See
//     https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
//     (ttng::WarpGroupDotOp corresponds to wgmma.fence followed by one or more
//     wgmma.async ops, so our understanding is that the two
//     ttng::WarpGroupDotOps don't have to correspond to wgmma.async ops with
//     the same shapes as specified in the docs, because there's an intervening
//     fence.)
//
// If the op can be properly async, this function returns the index of the dot
// in the loop's iter_args.  (Rule (2) above ensures this is well-defined.)
//
static std::optional<int> dotCanBeProperlyAsync(ttng::WarpGroupDotOp dotOp,
                                                scf::ForOp forOp) {
  LDBG("Considering whether to make MMAv3 dot properly async: " << dotOp);

  // Rule 1: All shmem operands are multi-buffered.
  auto checkOperand = [&](Value operand) {
    if (!isa<ttg::SharedEncodingTrait>(
            cast<ttg::TensorOrMemDesc>(operand.getType()).getEncoding())) {
      // Rule 1a: Register operands must not be modified within the loop.
      // First, check for chained WGMMA as an exception.
      if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(operand.getDefiningOp())) {
        return isa<ttg::NvidiaMmaEncodingAttr>(
            cvt.getSrc().getType().getEncoding());
      }
      // And then, do a stricter-than-necessary check for now, that the operand
      // is defined outside the loop.
      return forOp.isDefinedOutsideOfLoop(operand);
    }

    // If it's a shmem operand, it must either be defined outside the loop, or
    // come from an MemDescSubview op.  Only ConvertLayout and Trans ops are
    // allowed in between.
    Value transitiveOperand = operand;
    while (isa_and_nonnull<ttg::ConvertLayoutOp, ttg::MemDescTransOp>(
               transitiveOperand.getDefiningOp()) ||
           isa<BlockArgument>(transitiveOperand)) {
      auto blockArg = dyn_cast<BlockArgument>(transitiveOperand);
      if (blockArg && blockArg.getOwner() == forOp.getBody()) {
        transitiveOperand =
            cast<scf::YieldOp>(blockArg.getOwner()->getTerminator())
                .getOperand(blockArg.getArgNumber() - 1);
      } else if (Operation *def = transitiveOperand.getDefiningOp()) {
        transitiveOperand = def->getOperand(0);
      }
    }
    return forOp.isDefinedOutsideOfLoop(transitiveOperand) ||
           transitiveOperand.getDefiningOp<ttg::MemDescSubviewOp>();
  };

  // We don't have to call checkOperand on getC() because it's always in
  // registers, never in shmem.
  assert(isa<ttg::NvidiaMmaEncodingAttr>(dotOp.getC().getType().getEncoding()));
  if (!checkOperand(dotOp.getA()) || !checkOperand(dotOp.getB())) {
    LDBG("Can't make dot async because shmem operands aren't multi-buffered");
    return std::nullopt;
  }

  // Rule 2: The dot cannot be unconditionally used by any op in the loop.
  // Uses under `if` are allowed, as can be explicitly synced with a `wait 0`.
  int iterArgIdx = -1;
  Value iterArg = nullptr;
  SmallVector<std::pair<Operation *, int>> queue;
  for (auto &use : dotOp->getUses()) {
    queue.push_back({use.getOwner(), use.getOperandNumber()});
  }
  while (!queue.empty()) {
    auto [user, argIdx] = queue.pop_back_val();
    if (user->getParentOp() == forOp) {
      if (isa<scf::YieldOp>(user)) {
        if (iterArg) {
          // The dot is used by the loop's yield, but we can't have any other
          // uses.
          LDBG("Can't make dot async because dot is used by multiple ops in "
               "the loop.");
          return std::nullopt;
        }
        iterArgIdx = argIdx;
        iterArg = forOp.getRegionIterArg(argIdx);
        continue;
      }
      LDBG("Can't make dot async because dot is unconditionally used in the "
           "loop.");
      return std::nullopt;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp())) {
      if (isa<scf::YieldOp>(user)) {
        // The result is returned by the if, follow it further.
        auto uses = ifOp.getResult(argIdx).getUses();
        for (auto &use : uses) {
          queue.push_back({use.getOwner(), use.getOperandNumber()});
        }
      }
    } else {
      return std::nullopt;
    }
  }

  // Rule 3a: Are the only users of the dot's result from iteration i-1 other
  // MMAv3 dots?  If so, we're done, this dot can be properly async.
  if (llvm::all_of(iterArg.getUses(), [&](OpOperand &use) {
        return isa<ttng::WarpGroupDotOp>(use.getOwner()) &&
               use.getOperandNumber() == 2;
      })) {
    return iterArgIdx;
  }

  // Rule 3b: Are all users of the dot's result from iteration i-1 after the
  // first `warp_group_dot_wait {pendings=0}` op?  If so, the dot can be
  // properly async, but we have to thread its result from iteration i-1 through
  // the wait.
  auto waitOps = forOp.getBody()->getOps<ttng::WarpGroupDotWaitOp>();
  auto firstWaitOpIter = llvm::find_if(
      waitOps, [&](auto waitOp) { return waitOp.getPendings() == 0; });
  if (firstWaitOpIter != waitOps.end() &&
      llvm::all_of(iterArg.getUsers(), [&](Operation *user) {
        assert(forOp->isAncestor(user));
        while (user->getParentOp() != forOp) {
          user = user->getParentOp();
        }
        return (*firstWaitOpIter)->isBeforeInBlock(user);
      })) {
    LDBG("MMAv3 dot can be properly async because it follows a "
         "warp_group_dot_wait "
         "{pendings=0}.\n"
         << "  wait: " << *firstWaitOpIter << "\n"
         << "  dot: " << dotOp);
    threadValuesThroughWait(*firstWaitOpIter, {iterArg});
    return iterArgIdx;
  }

  LDBG("Can't make dot async because its result from i-1 is used by "
       "something other than another MMAv3 dot as the `c` operand.");
  return std::nullopt;
}

// If necessary, insert a dot-wait inside the loop, waiting for the results of
// the properly-async dots from iteration i-1 to complete.  (We pipeline to
// depth 2, so there are at most 2 copies of each warp_group_dot in flight at a
// time.)
//
// We can skip inserting the wait if we have a `warp_group_dot_wait
// {pendings=0}` somewhere in the loop.  To see why, consider:
//
//   warp_group_dot
//   warp_group_dot; wait 0  // synchronous dot
//   warp_group_dot
//   warp_group_dot
//
// In this example, there are three properly-async dots, so we'd normally put
// `wait 3` at the end of the loop, meaning "wait until there are 3 or fewer
// pending async dots".  But note that when this iteration of the loop
// completes, there are only *two* pending async dots from this iteration, so
// this wait would do nothing.  This is true in general, no matter where the
// `wait 0` appears.
static void insertAsyncWarpGroupDotWaitInLoop(
    scf::ForOp forOp,
    const llvm::MapVector<Operation *, int /*iterArgIdx*/> &properlyAsyncDots) {
  if (properlyAsyncDots.empty())
    return;

  if (llvm::any_of(forOp.getBody()->getOps<ttng::WarpGroupDotWaitOp>(),
                   [](auto wait) { return wait.getPendings() == 0; })) {
    return;
  }

  // Insert waits before the users of the properly async dots other than loop
  // yield.
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    SmallVector<OpOperand *> uses;
    for (auto &use : asyncDot->getUses()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
        continue;
      }
      uses.push_back(&use);
    }

    DenseMap<Block *, SmallVector<Value>> blockToUsers;
    for (auto use : uses) {
      auto block = use->getOwner()->getBlock();
      blockToUsers[block].push_back(use->get());
    }

    for (auto [block, users] : blockToUsers) {
      OpBuilder builder(block, block->begin());
      auto newWait = builder.create<ttng::WarpGroupDotWaitOp>(
          asyncDot->getLoc(), ArrayRef<Value>{}, 0);

      threadValuesThroughWait(newWait, users);
    }
  }

  // Add the wait right after the last properly-async dot.  This only needs to
  // wait for all properly-async dots from the i-1'th iteration to complete, IOW
  // we wait until there are most `asyncDots.size()` dots in flight.
  //
  // (You might want to put the wait at the end of the loop instead of right
  // after the last dot, but there could be a load into shmem between the last
  // async dot and the end of the loop, and that could clobber memory being used
  // by a dot.)
  IRRewriter builder(forOp.getContext());
  auto lastAsyncDot = properlyAsyncDots.back().first;
  builder.setInsertionPointAfter(lastAsyncDot);
  auto wait = builder.create<ttng::WarpGroupDotWaitOp>(
      lastAsyncDot->getLoc(),
      /*inputs=*/ArrayRef<Value>{}, properlyAsyncDots.size());

  // Thread the results of the async dots through the wait.
  SmallVector<Value> addlWaitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    addlWaitOperands.push_back(asyncDot->getResult(0));
  }
  threadValuesThroughWait(wait, addlWaitOperands);
}

// Convert MMAv3 ttng::WarpGroupDotOps {isAsync = False} (i.e. Hopper wgmma)
// into ttng::WarpGroupDotOps {isAsync = True} and insert
// ttng::WarpGroupDotWaitOps as necessary.
//
// We assume we have space for each dot to be pipelined to depth 2, i.e. each
// dot op in the loop can have at most 2 warp_group_dot ops in flight at once.
// (Each warp_group_dot op usually corresponds to a series of wgmma.async ops.)
void triton::asyncLaunchDots(scf::ForOp forOp) {
  LDBG("Original loop:\n" << *forOp);

  // First, change every MMAv3 ttng.warp_group_dot {isAsync=false}
  // into ttng.warp_group_dot {isAsync=true}.
  // The rest of this function is concerned with inserting
  // ttng.warp_group_dot_wait ops in the appropriate places.
  //
  // We call those dots that don't need to be followed immediately by a `wait 0`
  // "properly async", or sometimes just "async".
  //
  // For each dot, determine whether it can be properly async, or if it needs a
  // sync immediately after.  If it can be properly async, we know its only use
  // is in the loop's `yield` statement; asyncDots maps the op to its index in
  // the yield op.
  IRRewriter builder(forOp.getContext());
  llvm::MapVector<Operation *, int /*iterArgIdx*/> properlyAsyncDots;
  for (auto WarpGroupDotOp : forOp.getBody()->getOps<ttng::WarpGroupDotOp>()) {
    WarpGroupDotOp.setIsAsync(true);
    if (auto iterArgIdx = dotCanBeProperlyAsync(WarpGroupDotOp, forOp)) {
      properlyAsyncDots[WarpGroupDotOp] = *iterArgIdx;
    } else {
      builder.setInsertionPointAfter(WarpGroupDotOp);
      auto wait = builder.create<ttng::WarpGroupDotWaitOp>(
          WarpGroupDotOp.getLoc(), ArrayRef<Value>{},
          /*pendings=*/0);
      SmallVector<Value> waitOperands = {WarpGroupDotOp.getResult()};
      threadValuesThroughWait(wait, waitOperands);
    }
  }

  if (properlyAsyncDots.empty()) {
    LDBG("No properly async dots.");
    return;
  }

  // Next, insert a wait inside the loop.  We pipeline to depth 2, so the third
  // iteration's set of asynchronous dots (and their corresponding async copies
  // from global to shmem) can't start until the first iteration's set has
  // completed.
  insertAsyncWarpGroupDotWaitInLoop(forOp, properlyAsyncDots);

  // Finally, insert a wait after the loop, waiting for dots from the final
  // iteration of the loop.
  SmallVector<Value> waitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    waitOperands.push_back(forOp.getResult(iterArgIdx));
  }
  // Wait until there are 0 outstanding async dot ops.
  builder.setInsertionPointAfter(forOp);
  auto WarpGroupDotWaitAfterLoop = builder.create<ttng::WarpGroupDotWaitOp>(
      forOp.getLoc(), ArrayRef<Value>{}, 0);
  threadValuesThroughWait(WarpGroupDotWaitAfterLoop, waitOperands);
}
