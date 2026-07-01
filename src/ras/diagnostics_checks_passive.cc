/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <inttypes.h>
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "compiler.h"
#include "diagnostics.h"
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

// Emits a common incomplete-gather message for diagnostics checks.
static ncclResult_t rasDiagnosticsReportIncomplete(const struct rasDiagnosticsReporter* reporter, const char* checkName,
                                                   const struct rasDiagnosticsRankHeader* rank, int gatheredRanks) {
  char line[1024];

  snprintf(line, sizeof(line),
           "%s: diagnostics incomplete, gathered %d/%d ranks in comm 0x%lx/0x%lx/0x%lx "
           "(RAS overlay may not be ready)",
           checkName, gatheredRanks, rank->commNRanks, rank->commId.commHash, rank->commId.hostHash,
           rank->commId.pidHash);
  NCCLCHECK(reporter->emit(reporter->target, line));
  return ncclSuccess;
}

// *************************************************************************
// NCCL version check.
// *************************************************************************
struct rasDiagnosticsNcclVersionData {
  uint64_t versionCode;
};

static int rasDiagnosticsNcclVersionRecordCompare(const void* p1, const void* p2) {
  const struct rasDiagnosticsRankHeader* r1 = (const struct rasDiagnosticsRankHeader*)p1;
  const struct rasDiagnosticsRankHeader* r2 = (const struct rasDiagnosticsRankHeader*)p2;
  int cmp = rasDiagnosticsCommIdCompare(&r1->commId, &r2->commId);

  if (cmp != 0) return cmp;
  return (r1->commRank < r2->commRank ? -1 : (r1->commRank > r2->commRank ? 1 : 0));
}

static ncclResult_t rasDiagnosticsNcclVersionFillLocalData(const struct rasDiagnosticsCommSnapshot* comm,
                                                           void* checkData) {
  struct rasDiagnosticsNcclVersionData* versionData = (struct rasDiagnosticsNcclVersionData*)checkData;

  (void)comm; // unused
  versionData->versionCode = NCCL_VERSION_CODE;
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsNcclVersionCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                   struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsNcclVersionData),
                                              rasDiagnosticsNcclVersionFillLocalData, data));
  return ncclSuccess;
}

static const struct rasDiagnosticsNcclVersionData* rasDiagnosticsNcclVersionDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsNcclVersionData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

ncclResult_t rasDiagnosticsNcclVersionSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNcclVersionData));
  int nRecords;

  (void)ctx; // unused

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics NCCL version check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics NCCL version check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics NCCL version check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsNcclVersionRecordCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    uint64_t expectedVersionCode = rasDiagnosticsNcclVersionDataFromRecord(startRecord)->versionCode;
    int expectedRank = startRank->commRank;
    int commNRanks = startRank->commNRanks;
    uint64_t mismatchVersionCode = expectedVersionCode;
    int mismatchRank = expectedRank;
    bool foundMismatch = false;
    int end = start + 1;

    while (end < nRecords) {
      const char* record = records + end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      uint64_t versionCode = rasDiagnosticsNcclVersionDataFromRecord(record)->versionCode;

      if (rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      if (!foundMismatch && versionCode != expectedVersionCode) {
        foundMismatch = true;
        mismatchVersionCode = versionCode;
        mismatchRank = rank->commRank;
      }
      end++;
    }

    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "NCCL version", startRank, end - start), ret, exit);
    } else if (!foundMismatch) {
      char line[1024];
      snprintf(line, sizeof(line),
               "NCCL version: %" PRIu64 ".%" PRIu64 ".%" PRIu64
               " consistent across %d/%d ranks in comm 0x%lx/0x%lx/0x%lx",
               expectedVersionCode / 10000, (expectedVersionCode / 100) % 100, expectedVersionCode % 100, end - start,
               commNRanks, startRank->commId.commHash, startRank->commId.hostHash, startRank->commId.pidHash);
      NCCLCHECKGOTO(reporter->emit(reporter->target, line), ret, exit);
    } else {
      char line[1024];
      snprintf(line, sizeof(line),
               "NCCL version: mismatch across %d/%d ranks in comm 0x%lx/0x%lx/0x%lx "
               "rank %d has %" PRIu64 ", rank %d has %" PRIu64,
               end - start, commNRanks, startRank->commId.commHash, startRank->commId.hostHash,
               startRank->commId.pidHash, expectedRank, expectedVersionCode, mismatchRank, mismatchVersionCode);
      NCCLCHECKGOTO(reporter->emit(reporter->target, line), ret, exit);
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}
