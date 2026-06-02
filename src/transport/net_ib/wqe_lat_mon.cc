/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "wqe_lat_mon.h"
#include "common.h"

#include <math.h>
#include <mutex>
#include <stdint.h>
#include <string.h>
#include <time.h>

NCCL_PARAM(IbWqeLatencyThresholdNs, "IB_WQE_LATENCY_THRESHOLD_NS", 0);

bool ncclIbWqeLatEnabled = false;
uint64_t ncclIbWqeLatThresholdNs = 0;
uint64_t ncclIbWqeLatStallNs = 0;

static const uint64_t kWarnIntervalNs = 1000000000ULL;
static const uint64_t kStallFloorNs = 1000000ULL;
static const uint64_t kStallMultiple = 4;

static void ensureInitialized(void) {
  static std::once_flag once;
  std::call_once(once, []() {
    int64_t thr = ncclParamIbWqeLatencyThresholdNs();
    if (thr <= 0) return;
    ncclIbWqeLatThresholdNs = (uint64_t)thr;
    uint64_t stall = (uint64_t)thr * kStallMultiple;
    ncclIbWqeLatStallNs = stall < kStallFloorNs ? kStallFloorNs : stall;
    ncclIbWqeLatEnabled = true;
  });
}

uint64_t ncclIbWqeLatMonNowNs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void ncclIbWqeLatMonInit(struct ncclIbWqeLatMon* m) {
  ensureInitialized();
  memset(m, 0, sizeof(*m));
}

void ncclIbWqeLatMonStampSend(struct ncclIbQp* qp, struct ibv_send_wr* head) {
  struct ncclIbWqeLatMon* m = &qp->latMon;
  uint64_t now = 0;
  bool nowValid = false;
  for (struct ibv_send_wr* w = head; w != NULL; w = w->next) {
    if ((w->send_flags & IBV_SEND_SIGNALED) == 0) continue;
    if (!nowValid) {
      now = ncclIbWqeLatMonNowNs();
      nowValid = true;
    }
    if (!m->tracking) {
      m->tracking = true;
      m->trackedPostNs = now;
      m->pendingBefore = m->inflight;
    }
    m->inflight++;
  }
}

static void welfordUpdate(struct ncclIbWqeLatMon* m, uint64_t deltaNs) {
  m->count++;
  double x = (double)deltaNs;
  double delta = x - m->meanNs;
  m->meanNs += delta / (double)m->count;
  m->m2Ns += delta * (x - m->meanNs);
  if (deltaNs > m->maxNs) m->maxNs = deltaNs;
}

static void snapshotStats(const struct ncclIbWqeLatMon* m, struct ncclIbWqeLatStats* out) {
  if (!out) return;
  out->count = m->count;
  out->meanNs = m->meanNs;
  out->maxNs = m->maxNs;
  if (m->count > 1) {
    double var = m->m2Ns / (double)(m->count - 1);
    out->stddevNs = sqrt(var > 0.0 ? var : 0.0);
  } else {
    out->stddevNs = 0.0;
  }
}

bool ncclIbWqeLatMonOnComplete(struct ncclIbWqeLatMon* m, uint64_t tPollNs, uint64_t* outDeltaNs, uint64_t* outPostNs,
                               struct ncclIbWqeLatStats* outStats) {
  if (outDeltaNs) *outDeltaNs = 0;
  if (outPostNs) *outPostNs = 0;
  if (outStats) memset(outStats, 0, sizeof(*outStats));

  if (m->inflight > 0) m->inflight--;
  if (!m->tracking) return false;
  if (m->pendingBefore > 0) {
    m->pendingBefore--;
    return false;
  }

  uint64_t tPostNs = m->trackedPostNs;
  m->tracking = false;
  m->trackedPostNs = 0;

  uint64_t delta = (tPollNs > tPostNs) ? tPollNs - tPostNs : 0;
  welfordUpdate(m, delta);
  if (outDeltaNs) *outDeltaNs = delta;
  if (outPostNs) *outPostNs = tPostNs;
  snapshotStats(m, outStats);

  if (delta <= ncclIbWqeLatThresholdNs) return false;
  if (m->lastWarnNs && (tPollNs - m->lastWarnNs) < kWarnIntervalNs) return false;
  m->lastWarnNs = tPollNs;
  return true;
}

bool ncclIbWqeLatMonCheckStall(struct ncclIbWqeLatMon* m, uint64_t nowNs, uint64_t* outAgeNs, uint32_t* outInflight) {
  if (outAgeNs) *outAgeNs = 0;
  if (outInflight) *outInflight = 0;
  if (!m->tracking) return false;
  if (nowNs <= m->trackedPostNs) return false;

  uint64_t age = nowNs - m->trackedPostNs;
  if (outAgeNs) *outAgeNs = age;
  if (outInflight) *outInflight = m->inflight;

  if (age <= ncclIbWqeLatStallNs) return false;
  if (m->lastStallWarnNs && (nowNs - m->lastStallWarnNs) < kWarnIntervalNs) return false;
  m->lastStallWarnNs = nowNs;
  return true;
}

namespace {

struct peerInfo {
  char sock[SOCKET_NAME_MAXLEN + 1];
  char localGid[INET6_ADDRSTRLEN];
  char remoteGid[INET6_ADDRSTRLEN];
  const char* hca;
  uint16_t localLid;
  uint16_t remoteLid;
  uint8_t peerPort;
};

static void getPeerInfo(struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQp* qp, struct peerInfo* info) {
  memset(info, 0, sizeof(*info));
  info->hca = "?";
  union ncclSocketAddress addr;
  if (ncclSocketGetAddr(&base->sock, &addr) == ncclSuccess) {
    ncclSocketToString(&addr, info->sock);
  }
  struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(base, devIndex);
  if (devBase->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
    ibvGetGidStr(&devBase->gidInfo.localGid, info->localGid, sizeof(info->localGid));
    if (qp->remDevIdx >= 0 && qp->remDevIdx < base->nRemDevs) {
      ibvGetGidStr(&base->remDevs[qp->remDevIdx].remoteGid, info->remoteGid, sizeof(info->remoteGid));
    }
  }
  if (devBase->pd && devBase->pd->context && devBase->pd->context->device) {
    info->hca = devBase->pd->context->device->name;
  }
  info->localLid = (uint16_t)ncclIbDevs[devBase->ibDevN].portAttr.lid;
  info->remoteLid = (uint16_t)qp->rtrAttr.remoteLid;
  if (qp->remDevIdx >= 0 && qp->remDevIdx < base->nRemDevs) {
    info->peerPort = base->remDevs[qp->remDevIdx].ib_port;
  }
}

static void reportSlow(struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQp* qp, uint32_t qpn,
                       uint64_t deltaNs, uint64_t tPostNs, uint64_t tPollNs, const struct ncclIbWqeLatStats* s) {
  struct peerInfo p;
  getPeerInfo(base, devIndex, qp, &p);
  WARN("NET/IB: WQE slow: peer=%s | local: qpn=%u dev=%s port=%u lid=%u localGid=%s | "
       "remote: qpn=%u port=%u lid=%u remoteGid=%s | "
       "thr=%luns delta=%luns mean=%.0fns stddev=%.0fns max=%luns count=%lu | "
       "t_post=%lu t_poll=%lu",
       p.sock, qpn, p.hca, qp->rtrAttr.localIbPort, (unsigned)p.localLid, p.localGid, qp->rtrAttr.remoteQpNum,
       (unsigned)p.peerPort, (unsigned)p.remoteLid, p.remoteGid, (unsigned long)ncclIbWqeLatThresholdNs,
       (unsigned long)deltaNs, s->meanNs, s->stddevNs, (unsigned long)s->maxNs, (unsigned long)s->count,
       (unsigned long)tPostNs, (unsigned long)tPollNs);
}

static void reportStall(struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQp* qp, uint64_t ageNs,
                        uint32_t inflight) {
  struct peerInfo p;
  getPeerInfo(base, devIndex, qp, &p);
  WARN("NET/IB: WQE stall (no CQE): peer=%s | local: qpn=%u dev=%s port=%u lid=%u localGid=%s | "
       "remote: qpn=%u port=%u lid=%u remoteGid=%s | "
       "stall_thr=%luns age=%luns inflight=%u",
       p.sock, qp->qp->qp_num, p.hca, qp->rtrAttr.localIbPort, (unsigned)p.localLid, p.localGid,
       qp->rtrAttr.remoteQpNum, (unsigned)p.peerPort, (unsigned)p.remoteLid, p.remoteGid,
       (unsigned long)ncclIbWqeLatStallNs, (unsigned long)ageNs, inflight);
}

}  // namespace

void ncclIbWqeLatHandleCompletion(struct ncclIbNetCommBase* base, int devIndex, struct ibv_wc* wc, uint64_t* tPollNs,
                                  bool* tPollNsValid) {
  const bool isSendOpcode =
    (wc->opcode == IBV_WC_SEND) || (wc->opcode == IBV_WC_RDMA_WRITE) || (wc->opcode == IBV_WC_RDMA_READ);
  if (!isSendOpcode) return;
  struct ncclIbQp* qp = NULL;
  int qpIndex = -1;
  if (ncclIbCommBaseGetQpByQpNum(base, devIndex, wc->qp_num, &qp, &qpIndex) != ncclSuccess) return;
  if (qp == NULL) return;
  if (!*tPollNsValid) {
    *tPollNs = ncclIbWqeLatMonNowNs();
    *tPollNsValid = true;
  }
  uint64_t deltaNs = 0, tPostNs = 0;
  struct ncclIbWqeLatStats stats;
  if (ncclIbWqeLatMonOnComplete(&qp->latMon, *tPollNs, &deltaNs, &tPostNs, &stats)) {
    reportSlow(base, devIndex, qp, wc->qp_num, deltaNs, tPostNs, *tPollNs, &stats);
  }
}

void ncclIbWqeLatScanStalls(struct ncclIbNetCommBase* base, int devIndex) {
  if (base->nqps <= 0 || base->vProps.ndevs <= 0) return;
  int nqpsPerDev = base->nqps / base->vProps.ndevs;
  uint64_t now = ncclIbWqeLatMonNowNs();
  for (int k = 0; k < nqpsPerDev; k++) {
    struct ncclIbQp* qp = &base->qps[base->vProps.ndevs * k + devIndex];
    uint64_t age = 0;
    uint32_t inflight = 0;
    if (ncclIbWqeLatMonCheckStall(&qp->latMon, now, &age, &inflight)) {
      reportStall(base, devIndex, qp, age, inflight);
    }
  }
}
