/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "compiler.h"
#include "cudawrap.h"
#include "diagnostics.h"
#include "nvmlwrap.h"
#include "param.h"
#include "transport.h"

// Generic local payload helpers shared by all diagnostics checks.
// A local payload is a flat array of records:
//   [rasDiagnosticsRankHeader][check-specific data][padding]
// RAS_COLL_DIAG gathers variable-size peer payloads, so different peers may
// contribute a different number of records for the same check.

typedef ncclResult_t (*rasDiagnosticsFillLocalDataFn)(const struct rasDiagnosticsCommSnapshot* comm, void* checkData);

static void rasDiagnosticsCommSnapshotInit(struct rasDiagnosticsCommSnapshot* snapshot, const struct ncclComm* comm) {
  snapshot->rank.commId.commHash = comm->commHash;
  snapshot->rank.commId.hostHash = comm->peerInfo[0].hostHash;
  snapshot->rank.commId.pidHash = comm->peerInfo[0].pidHash;
  snapshot->rank.commRank = comm->rank;
  snapshot->rank.commNRanks = comm->nRanks;
  snapshot->cudaDev = comm->cudaDev;
  snapshot->nvmlDev = comm->nvmlDev;
  snapshot->busId = comm->busId;
  snapshot->localRank = comm->localRank;
  snapshot->localRanks = comm->localRanks;
}

static int rasDiagnosticsCommIdCompare(const struct rasCommId* id1, const struct rasCommId* id2) {
  if (id1->commHash != id2->commHash) return (id1->commHash < id2->commHash ? -1 : 1);
  if (id1->hostHash != id2->hostHash) return (id1->hostHash < id2->hostHash ? -1 : 1);
  return (id1->pidHash < id2->pidHash ? -1 : (id1->pidHash > id2->pidHash ? 1 : 0));
}

static bool rasDiagnosticsCommMatchesContext(const struct rasDiagnosticsContext* ctx, const struct ncclComm* comm) {
  struct rasCommId commId;

  if (!ctx->hasCommFilter) return true;
  commId.commHash = comm->commHash;
  commId.hostHash = comm->peerInfo[0].hostHash;
  commId.pidHash = comm->peerInfo[0].pidHash;
  return rasDiagnosticsCommIdCompare(&commId, &ctx->commFilter) == 0;
}

static size_t rasDiagnosticsLocalRecordStride(size_t checkDataSize) {
  const size_t align = alignof(struct rasDiagnosticsRankHeader);
  size_t recordSize = sizeof(struct rasDiagnosticsRankHeader) + checkDataSize;

  return ((recordSize + align - 1) / align) * align;
}

static ncclResult_t rasDiagnosticsFillLocalRecord(char* record, const struct rasDiagnosticsCommSnapshot* snapshot,
                                                  rasDiagnosticsFillLocalDataFn fillCheckData) {
  memcpy(record, &snapshot->rank, sizeof(snapshot->rank));

  NCCLCHECK(fillCheckData(snapshot, record + sizeof(struct rasDiagnosticsRankHeader)));
  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsCollectLocalRecords(const struct rasDiagnosticsContext* ctx, size_t checkDataSize,
                                                      rasDiagnosticsFillLocalDataFn fillCheckData,
                                                      struct rasDiagnosticsLocalData* data) {
  ncclUniquePtr<struct rasDiagnosticsCommSnapshot> snapshots;
  ncclUniquePtr<char> records;
  size_t recordStride;
  int nRecords = 0;
  size_t nBytes;

  if (data == nullptr) {
    WARN("RAS diagnostics check local data output is null");
    return ncclInternalError;
  }
  memset(data, 0, sizeof(*data));

  if (ctx == nullptr) {
    WARN("RAS diagnostics check local collection requested with null context");
    return ncclInternalError;
  }
  if (fillCheckData == nullptr || checkDataSize > (size_t)INT_MAX - sizeof(struct rasDiagnosticsRankHeader)) {
    WARN("RAS diagnostics check local data size %zu is invalid", checkDataSize);
    return ncclInternalError;
  }

  recordStride = rasDiagnosticsLocalRecordStride(checkDataSize);
  if (recordStride > (size_t)INT_MAX) {
    WARN("RAS diagnostics check local record stride %zu is invalid", recordStride);
    return ncclInternalError;
  }

  {
    std::lock_guard<std::mutex> lock(ncclCommsMutex);

    for (int i = 0; i < nNcclComms; i++) {
      struct ncclComm* comm = ncclComms[i];
      if (comm == nullptr) continue;
      if (!COMPILER_ATOMIC_LOAD(&comm->peerInfoValid, std::memory_order_acquire)) continue;
      if (!rasDiagnosticsCommMatchesContext(ctx, comm)) continue;
      nRecords++;
    }

    if (nRecords == 0) return ncclSuccess;
    if ((size_t)nRecords > (size_t)INT_MAX / recordStride) {
      WARN("RAS diagnostics check local data too large");
      return ncclInternalError;
    }

    NCCLCHECK(ncclCalloc(snapshots, nRecords));
    for (int i = 0, recordIdx = 0; i < nNcclComms && recordIdx < nRecords; i++) {
      struct ncclComm* comm = ncclComms[i];
      if (comm == nullptr) continue;
      if (!COMPILER_ATOMIC_LOAD(&comm->peerInfoValid, std::memory_order_acquire)) continue;
      if (!rasDiagnosticsCommMatchesContext(ctx, comm)) continue;
      rasDiagnosticsCommSnapshotInit(snapshots.get() + recordIdx, comm);
      recordIdx++;
    }
  }

  nBytes = (size_t)nRecords * recordStride;
  NCCLCHECK(ncclCalloc(records, nBytes));

  // Check-specific probes consume snapshots after releasing ncclCommsMutex.
  for (int recordIdx = 0; recordIdx < nRecords; recordIdx++) {
    NCCLCHECK(rasDiagnosticsFillLocalRecord(records.get() + recordIdx * recordStride, snapshots.get() + recordIdx,
                                            fillCheckData));
  }

  data->records = records.release();
  data->recordsBytes = (int)nBytes;
  data->recordStride = (int)recordStride;
  data->nRecords = nRecords;
  return ncclSuccess;
}

static const struct rasDiagnosticsRankHeader* rasDiagnosticsRankHeaderFromRecord(const char* record) {
  return (const struct rasDiagnosticsRankHeader*)record;
}

// Orders records by comm, then rank; depends only on the rank header, so all checks share it.
static int rasDiagnosticsRankHeaderCompare(const void* p1, const void* p2) {
  const struct rasDiagnosticsRankHeader* r1 = (const struct rasDiagnosticsRankHeader*)p1;
  const struct rasDiagnosticsRankHeader* r2 = (const struct rasDiagnosticsRankHeader*)p2;
  int cmp = rasDiagnosticsCommIdCompare(&r1->commId, &r2->commId);

  if (cmp != 0) return cmp;
  return (r1->commRank < r2->commRank ? -1 : (r1->commRank > r2->commRank ? 1 : 0));
}

// Number of individual ranks shown in a mismatch set before it is truncated with a total count.
#define RAS_DIAG_RANK_SET_MAX 8

// Formats a set of ranks as "{a,b,c}", or "{a,b,...} (N=total)" once the set exceeds RAS_DIAG_RANK_SET_MAX.
static void rasDiagnosticsFormatRankSet(char* buf, size_t bufLen, const int* ranks, int nStored, int nTotal) {
  int show = nStored < RAS_DIAG_RANK_SET_MAX ? nStored : RAS_DIAG_RANK_SET_MAX;
  int pos = snprintf(buf, bufLen, "{");

  for (int i = 0; i < show && pos > 0 && (size_t)pos < bufLen; i++) {
    pos += snprintf(buf + pos, bufLen - pos, "%s%d", i == 0 ? "" : ",", ranks[i]);
  }
  if (pos > 0 && (size_t)pos < bufLen) {
    if (nTotal > show) snprintf(buf + pos, bufLen - pos, ",...} (N=%d)", nTotal);
    else snprintf(buf + pos, bufLen - pos, "}");
  }
}

// Severity tags, space-padded so message text aligns across severities.
#define RAS_DIAG_TAG_OK "[OK]   "
#define RAS_DIAG_TAG_INFO "[INFO] "

// Formats a diagnostics line (severity tag + message) and emits it through the (already validated) reporter.
static ncclResult_t __attribute__((format(printf, 3, 4))) rasDiagnosticsReport(
  const struct rasDiagnosticsReporter* reporter, const char* tag, const char* fmt, ...) {
  char line[1024];
  int pos = snprintf(line, sizeof(line), "%s", tag);
  if (pos < 0 || (size_t)pos >= sizeof(line)) {
    WARN("RAS diagnostics report formatting failed");
    return ncclInternalError;
  }

  va_list args;
  va_start(args, fmt);
  vsnprintf(line + pos, sizeof(line) - (size_t)pos, fmt, args);
  va_end(args);
  return reporter->emit(reporter->target, line);
}

static ncclResult_t rasDiagnosticsReportIncomplete(const struct rasDiagnosticsReporter* reporter, const char* checkName,
                                                   const struct rasDiagnosticsRankHeader* rank, int gatheredRanks) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "%s: diagnostics incomplete, gathered %d/%d ranks in comm 0x%lx/0x%lx/0x%lx "
                              "(RAS overlay may not be ready)",
                              checkName, gatheredRanks, rank->commNRanks, rank->commId.commHash, rank->commId.hostHash,
                              rank->commId.pidHash);
}

// *************************************************************************
// GPU model and count consistency check.
// *************************************************************************
#define RAS_DIAG_GPU_MODEL_NAME_LEN NVML_DEVICE_NAME_BUFFER_SIZE
#define RAS_DIAG_GPU_MODEL_UNKNOWN "unknown"

struct rasDiagnosticsGpuModelData {
  uint8_t nGpus;
  char model[RAS_DIAG_GPU_MODEL_NAME_LEN];
};

static ncclResult_t rasDiagnosticsGpuModelFillLocalData(const struct rasDiagnosticsCommSnapshot* comm,
                                                        void* checkData) {
  struct rasDiagnosticsGpuModelData* gpuData = (struct rasDiagnosticsGpuModelData*)checkData;

  unsigned int nDev = 0;

  gpuData->nGpus = 0;
  gpuData->model[0] = '\0';

  if (ncclNvmlDeviceGetCount(&nDev) == ncclSuccess) gpuData->nGpus = (uint8_t)nDev;
  if (comm->nvmlDev >= 0 && comm->nvmlDev < ncclNvmlDeviceCount) {
    nvmlDevice_t device;
    if (ncclNvmlDeviceGetHandleByIndex((unsigned int)comm->nvmlDev, &device) == ncclSuccess) {
      if (ncclNvmlDeviceGetName(device, gpuData->model, sizeof(gpuData->model)) != ncclSuccess) {
        gpuData->model[0] = '\0';
      }
    }
  }

  if (gpuData->model[0] == '\0') snprintf(gpuData->model, sizeof(gpuData->model), "%s", RAS_DIAG_GPU_MODEL_UNKNOWN);
  gpuData->model[sizeof(gpuData->model) - 1] = '\0';
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsGpuModelCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsGpuModelData),
                                              rasDiagnosticsGpuModelFillLocalData, data));
  return ncclSuccess;
}

static const struct rasDiagnosticsGpuModelData* rasDiagnosticsGpuModelDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsGpuModelData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

ncclResult_t rasDiagnosticsGpuModelSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsGpuModelData));
  int nRecords;

  (void)ctx;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics GPU model check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics GPU model check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics GPU model check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    const struct rasDiagnosticsGpuModelData* startData = rasDiagnosticsGpuModelDataFromRecord(startRecord);
    int expectedRank = startRank->commRank;
    int commNRanks = startRank->commNRanks;
    int expectedNGpus = startData->nGpus;
    int countMismatchRanks[RAS_DIAG_RANK_SET_MAX];
    int modelMismatchRanks[RAS_DIAG_RANK_SET_MAX];
    int nCountMismatchStored = 0, nCountMismatch = 0;
    int nModelMismatchStored = 0, nModelMismatch = 0;
    const char* expectedModel = startData->model;
    int end = start + 1;

    while (end < nRecords) {
      const char* record = records + end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsGpuModelData* gpuData = rasDiagnosticsGpuModelDataFromRecord(record);

      if (rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      if (gpuData->nGpus != expectedNGpus) {
        if (nCountMismatchStored < RAS_DIAG_RANK_SET_MAX) countMismatchRanks[nCountMismatchStored++] = rank->commRank;
        nCountMismatch++;
      }
      if (strncmp(gpuData->model, expectedModel, RAS_DIAG_GPU_MODEL_NAME_LEN) != 0) {
        if (nModelMismatchStored < RAS_DIAG_RANK_SET_MAX) modelMismatchRanks[nModelMismatchStored++] = rank->commRank;
        nModelMismatch++;
      }
      end++;
    }

    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "GPU inventory", startRank, end - start), ret, exit);
    } else if (nCountMismatch == 0 && nModelMismatch == 0) {
      bool countKnown = expectedNGpus > 0;
      bool modelKnown = strncmp(expectedModel, RAS_DIAG_GPU_MODEL_UNKNOWN, RAS_DIAG_GPU_MODEL_NAME_LEN) != 0;
      if (countKnown && modelKnown) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK,
                                           "GPU inventory: %dx %s per node consistent across %d ranks in comm 0x%lx",
                                           expectedNGpus, expectedModel, commNRanks, startRank->commId.commHash),
                      ret, exit);
      } else if (countKnown) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "GPU inventory: %d GPUs per node consistent across %d ranks in comm 0x%lx, "
                                           "GPU model unavailable via NVML",
                                           expectedNGpus, commNRanks, startRank->commId.commHash),
                      ret, exit);
      } else if (modelKnown) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "GPU inventory: %s per node consistent across %d ranks in comm 0x%lx, "
                                           "GPU count unavailable via NVML",
                                           expectedModel, commNRanks, startRank->commId.commHash),
                      ret, exit);
      } else {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "GPU inventory: unavailable via NVML across %d ranks in comm 0x%lx",
                                           commNRanks, startRank->commId.commHash),
                      ret, exit);
      }
    } else {
      if (nCountMismatch > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), countMismatchRanks, nCountMismatchStored, nCountMismatch);
        NCCLCHECKGOTO(
          rasDiagnosticsReport(
            reporter, RAS_DIAG_TAG_INFO,
            "GPU inventory: count mismatch across %d ranks in comm 0x%lx, rank(s) %s differ from rank %d (%d)",
            commNRanks, startRank->commId.commHash, rankSet, expectedRank, expectedNGpus),
          ret, exit);
      }
      if (nModelMismatch > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), modelMismatchRanks, nModelMismatchStored, nModelMismatch);
        NCCLCHECKGOTO(
          rasDiagnosticsReport(
            reporter, RAS_DIAG_TAG_INFO,
            "GPU inventory: model mismatch across %d ranks in comm 0x%lx, rank(s) %s differ from rank %d (%s)",
            commNRanks, startRank->commId.commHash, rankSet, expectedRank, expectedModel),
          ret, exit);
      }
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}

// *************************************************************************
// CUDA driver version consistency check.
// *************************************************************************
#define RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN 0

struct rasDiagnosticsCudaDriverVersionData {
  uint32_t version;
};

static const char* rasDiagnosticsCudaDriverVersionString(uint32_t version, char* buf, size_t bufLen) {
  if (version == RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN) snprintf(buf, bufLen, "unavailable");
  else snprintf(buf, bufLen, "%u", version);
  return buf;
}

static ncclResult_t rasDiagnosticsCudaDriverVersionFillLocalData(const struct rasDiagnosticsCommSnapshot* comm,
                                                                 void* checkData) {
  struct rasDiagnosticsCudaDriverVersionData* versionData = (struct rasDiagnosticsCudaDriverVersionData*)checkData;
  int version = 0;

  (void)comm;
  versionData->version = RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN;
  if (ncclCudaDriverVersion(&version) == ncclSuccess && version > 0) {
    versionData->version = (uint32_t)version;
  }
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsCudaDriverVersionCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                         struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsCudaDriverVersionData),
                                              rasDiagnosticsCudaDriverVersionFillLocalData, data));
  return ncclSuccess;
}

static const struct rasDiagnosticsCudaDriverVersionData* rasDiagnosticsCudaDriverVersionDataFromRecord(
  const char* record) {
  return (const struct rasDiagnosticsCudaDriverVersionData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

ncclResult_t rasDiagnosticsCudaDriverVersionSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsCudaDriverVersionData));
  int nRecords;

  (void)ctx;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics CUDA driver version check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics CUDA driver version check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics CUDA driver version check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    uint32_t expectedVersion = rasDiagnosticsCudaDriverVersionDataFromRecord(startRecord)->version;
    int expectedRank = startRank->commRank;
    int commNRanks = startRank->commNRanks;
    int mismatchRanks[RAS_DIAG_RANK_SET_MAX];
    int nMismatchStored = 0, nMismatch = 0;
    int end = start + 1;

    while (end < nRecords) {
      const char* record = records + end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      uint32_t version = rasDiagnosticsCudaDriverVersionDataFromRecord(record)->version;

      if (rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      if (version != expectedVersion) {
        if (nMismatchStored < RAS_DIAG_RANK_SET_MAX) mismatchRanks[nMismatchStored++] = rank->commRank;
        nMismatch++;
      }
      end++;
    }

    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "CUDA driver version", startRank, end - start), ret, exit);
    } else if (nMismatch == 0) {
      char version[32];
      if (expectedVersion == RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "CUDA driver version: unavailable across %d ranks in comm 0x%lx", commNRanks,
                                           startRank->commId.commHash),
                      ret, exit);
      } else {
        NCCLCHECKGOTO(rasDiagnosticsReport(
                        reporter, RAS_DIAG_TAG_OK, "CUDA driver version: %s consistent across %d ranks in comm 0x%lx",
                        rasDiagnosticsCudaDriverVersionString(expectedVersion, version, sizeof(version)), commNRanks,
                        startRank->commId.commHash),
                      ret, exit);
      }
    } else {
      char rankSet[128];
      char version[32];
      rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), mismatchRanks, nMismatchStored, nMismatch);
      NCCLCHECKGOTO(
        rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                             "CUDA driver version: mismatch across %d ranks in comm 0x%lx, "
                             "rank(s) %s differ from rank %d (%s)",
                             commNRanks, startRank->commId.commHash, rankSet, expectedRank,
                             rasDiagnosticsCudaDriverVersionString(expectedVersion, version, sizeof(version))),
        ret, exit);
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}

// *************************************************************************
// Volatile uncorrected ECC counter check.
// *************************************************************************
// Uncorrected volatile ECC errors are always flagged; corrected errors are only flagged once they reach
// NCCL_DIAGNOSTICS_ECC_THRESHOLD (0 disables corrected reporting).
NCCL_PARAM(DiagnosticsEccThreshold, "DIAGNOSTICS_ECC_THRESHOLD", 0);

struct rasDiagnosticsEccData {
  uint64_t correctedSram;
  uint64_t uncorrectedSram;
  uint64_t correctedDram;
  uint64_t uncorrectedDram;
  uint8_t available; // 1 only if NVML reported every ECC counter for this rank; 0 otherwise.
};

// Reads one volatile ECC counter. Returns true only if NVML answered.
static bool rasDiagnosticsEccReadCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType,
                                         nvmlMemoryLocation_t location, uint64_t* out) {
  unsigned long long value = 0;
  if (ncclNvmlDeviceGetMemoryErrorCounter(device, errorType, NVML_VOLATILE_ECC, location, &value) != ncclSuccess)
    return false;
  *out = value;
  return true;
}

static ncclResult_t rasDiagnosticsEccFillLocalData(const struct rasDiagnosticsCommSnapshot* comm, void* checkData) {
  struct rasDiagnosticsEccData* eccData = (struct rasDiagnosticsEccData*)checkData;

  eccData->correctedSram = 0;
  eccData->uncorrectedSram = 0;
  eccData->correctedDram = 0;
  eccData->uncorrectedDram = 0;
  eccData->available = 0;

  if (comm->nvmlDev >= 0 && comm->nvmlDev < ncclNvmlDeviceCount) {
    nvmlDevice_t device;
    if (ncclNvmlDeviceGetHandleByIndex((unsigned int)comm->nvmlDev, &device) == ncclSuccess) {
      // Available only if NVML answered every counter, so a failed read is never reported as a zero.
      bool ok = true;
      ok &= rasDiagnosticsEccReadCounter(device, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_MEMORY_LOCATION_SRAM,
                                         &eccData->correctedSram);
      ok &= rasDiagnosticsEccReadCounter(device, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_MEMORY_LOCATION_SRAM,
                                         &eccData->uncorrectedSram);
      ok &= rasDiagnosticsEccReadCounter(device, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_MEMORY_LOCATION_DRAM,
                                         &eccData->correctedDram);
      ok &= rasDiagnosticsEccReadCounter(device, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_MEMORY_LOCATION_DRAM,
                                         &eccData->uncorrectedDram);
      eccData->available = ok ? 1 : 0;
    }
  }
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsEccCollectLocal(const struct rasDiagnosticsContext* ctx,
                                           struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsEccData), rasDiagnosticsEccFillLocalData,
                                              data));
  return ncclSuccess;
}

static const struct rasDiagnosticsEccData* rasDiagnosticsEccDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsEccData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

ncclResult_t rasDiagnosticsEccSummarize(const struct rasDiagnosticsContext* ctx,
                                        const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsEccData));
  const unsigned long long threshold = (unsigned long long)ncclParamDiagnosticsEccThreshold();
  int nRecords;

  (void)ctx;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics ECC check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics ECC check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics ECC check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    int commNRanks = startRank->commNRanks;
    int uncorrectedRanks[RAS_DIAG_RANK_SET_MAX];
    int correctedRanks[RAS_DIAG_RANK_SET_MAX];
    int nUncorrectedStored = 0, nUncorrected = 0;
    int nCorrectedStored = 0, nCorrected = 0;
    int nAvailable = 0;
    unsigned long long worstUncorrected = 0, worstCorrected = 0;
    int end = start;

    while (end < nRecords) {
      const char* record = records + end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsEccData* eccData;
      unsigned long long uncorrected, corrected;

      if (end > start && rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      eccData = rasDiagnosticsEccDataFromRecord(record);
      if (eccData->available) nAvailable++;
      uncorrected = eccData->uncorrectedSram + eccData->uncorrectedDram;
      corrected = eccData->correctedSram + eccData->correctedDram;
      if (uncorrected > 0) {
        if (nUncorrectedStored < RAS_DIAG_RANK_SET_MAX) uncorrectedRanks[nUncorrectedStored++] = rank->commRank;
        nUncorrected++;
        if (uncorrected > worstUncorrected) worstUncorrected = uncorrected;
      }
      if (threshold > 0 && corrected >= threshold) {
        if (nCorrectedStored < RAS_DIAG_RANK_SET_MAX) correctedRanks[nCorrectedStored++] = rank->commRank;
        nCorrected++;
        if (corrected > worstCorrected) worstCorrected = corrected;
      }
      end++;
    }

    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "ECC", startRank, end - start), ret, exit);
    } else if (nAvailable == 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ECC: unavailable via NVML across %d ranks in comm 0x%lx", commNRanks,
                                         startRank->commId.commHash),
                    ret, exit);
    } else if (nUncorrected == 0 && nCorrected == 0) {
      if (nAvailable == commNRanks) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK,
                                           "ECC: no uncorrected volatile errors across %d ranks in comm 0x%lx",
                                           commNRanks, startRank->commId.commHash),
                      ret, exit);
      } else {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "ECC: no uncorrected volatile errors across %d of %d ranks in comm 0x%lx "
                                           "(ECC counters unavailable via NVML on %d ranks)",
                                           nAvailable, commNRanks, startRank->commId.commHash, commNRanks - nAvailable),
                      ret, exit);
      }
    } else {
      if (nUncorrected > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), uncorrectedRanks, nUncorrectedStored, nUncorrected);
        NCCLCHECKGOTO(
          rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                               "ECC: uncorrected volatile errors on rank(s) %s (worst=%llu) across %d ranks "
                               "in comm 0x%lx",
                               rankSet, worstUncorrected, commNRanks, startRank->commId.commHash),
          ret, exit);
      }
      if (nCorrected > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), correctedRanks, nCorrectedStored, nCorrected);
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "ECC: corrected volatile errors at or above threshold %llu on rank(s) %s "
                                           "(worst=%llu) across %d ranks in comm 0x%lx",
                                           threshold, rankSet, worstCorrected, commNRanks, startRank->commId.commHash),
                      ret, exit);
      }
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}

// *************************************************************************
// Per-NVLink operational state check.
// *************************************************************************
// Counts each device's valid NVLinks and how many are not enabled. PCIe-only devices report zero links and are
// skipped silently.

// NVML has no valid-link count query, so scan link IDs [0, RAS_DIAG_NVLINK_MAX_LINKS) and keep the ones it marks valid.
#define RAS_DIAG_NVLINK_MAX_LINKS 18

struct rasDiagnosticsNvLinkData {
  uint8_t nLinks; // Valid NVLinks reported by NVML.
  uint8_t nInactive; // Valid NVLinks not in the NVML_FEATURE_ENABLED state.
};

static ncclResult_t rasDiagnosticsNvLinkFillLocalData(const struct rasDiagnosticsCommSnapshot* comm, void* checkData) {
  struct rasDiagnosticsNvLinkData* nvlData = (struct rasDiagnosticsNvLinkData*)checkData;

  nvlData->nLinks = 0;
  nvlData->nInactive = 0;

  if (comm->nvmlDev >= 0 && comm->nvmlDev < ncclNvmlDeviceCount) {
    nvmlDevice_t device;
    if (ncclNvmlDeviceGetHandleByIndex((unsigned int)comm->nvmlDev, &device) == ncclSuccess) {
      for (unsigned int link = 0; link < RAS_DIAG_NVLINK_MAX_LINKS; link++) {
        unsigned int valid = 0;
        if (ncclNvmlDeviceGetNvLinkCapability(device, link, NVML_NVLINK_CAP_VALID, &valid) != ncclSuccess || valid == 0)
          continue;
        nvlData->nLinks++;
        // Unreadable state counts as inactive rather than assumed healthy.
        nvmlEnableState_t state = NVML_FEATURE_DISABLED;
        if (ncclNvmlDeviceGetNvLinkState(device, link, &state) != ncclSuccess || state != NVML_FEATURE_ENABLED)
          nvlData->nInactive++;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsNvLinkCollectLocal(const struct rasDiagnosticsContext* ctx,
                                              struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsNvLinkData),
                                              rasDiagnosticsNvLinkFillLocalData, data));
  return ncclSuccess;
}

static const struct rasDiagnosticsNvLinkData* rasDiagnosticsNvLinkDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsNvLinkData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

ncclResult_t rasDiagnosticsNvLinkSummarize(const struct rasDiagnosticsContext* ctx,
                                           const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNvLinkData));
  int nRecords;

  (void)ctx;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics NVLink check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics NVLink check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics NVLink check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    // Sorted by rank, so the group's first record is its lowest rank; use it as the reference.
    int refLinks = rasDiagnosticsNvLinkDataFromRecord(startRecord)->nLinks;
    int commNRanks = startRank->commNRanks;
    int countMismatchRanks[RAS_DIAG_RANK_SET_MAX];
    int inactiveRanks[RAS_DIAG_RANK_SET_MAX];
    int nCountMismatchStored = 0, nCountMismatch = 0;
    int nInactiveStored = 0, nInactive = 0;
    int nWithLinks = 0;
    int end = start;

    while (end < nRecords) {
      const char* record = records + end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsNvLinkData* nvlData;

      if (end > start && rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      nvlData = rasDiagnosticsNvLinkDataFromRecord(record);
      if (nvlData->nLinks > 0) nWithLinks++;
      if (nvlData->nLinks != refLinks) {
        if (nCountMismatchStored < RAS_DIAG_RANK_SET_MAX) countMismatchRanks[nCountMismatchStored++] = rank->commRank;
        nCountMismatch++;
      }
      if (nvlData->nInactive > 0) {
        if (nInactiveStored < RAS_DIAG_RANK_SET_MAX) inactiveRanks[nInactiveStored++] = rank->commRank;
        nInactive++;
      }
      end++;
    }

    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "NVLink", startRank, end - start), ret, exit);
    } else if (nWithLinks == 0) {
      // No device exposes NVLink (e.g. PCIe-only); nothing to report.
    } else if (nCountMismatch == 0 && nInactive == 0) {
      NCCLCHECKGOTO(
        rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK,
                             "NVLink: found %d link(s) per device, all active across %d ranks in comm 0x%lx", refLinks,
                             commNRanks, startRank->commId.commHash),
        ret, exit);
    } else {
      if (nCountMismatch > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), countMismatchRanks, nCountMismatchStored, nCountMismatch);
        NCCLCHECKGOTO(
          rasDiagnosticsReport(
            reporter, RAS_DIAG_TAG_INFO,
            "NVLink: link-count mismatch across %d ranks in comm 0x%lx, rank(s) %s differ from rank %d (%d)",
            commNRanks, startRank->commId.commHash, rankSet, startRank->commRank, refLinks),
          ret, exit);
      }
      if (nInactive > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), inactiveRanks, nInactiveStored, nInactive);
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "NVLink: inactive link(s) on rank(s) %s across %d ranks in comm 0x%lx",
                                           rankSet, commNRanks, startRank->commId.commHash),
                      ret, exit);
      }
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}
