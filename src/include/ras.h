/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_RAS_H_
#define NCCL_RAS_H_

#include "socket.h"

// Structure used to communicate data about NCCL ranks from NCCL threads to RAS.
struct rasRankInit {
  union ncclSocketAddress addr;
  ncclPid_t pid;
  int cudaDev;
  int nvmlDev;
  uint64_t hostHash;
  uint64_t pidHash;
};

typedef enum {
  NCCL_RAS_DIAG_OFF = 0,
  NCCL_RAS_DIAG_PASSIVE = 1,
  NCCL_RAS_DIAG_ACTIVE = 2,
} ncclRasDiagMode;

ncclResult_t ncclRasCommInit(struct ncclComm* comm, struct rasRankInit* myRank);
ncclResult_t ncclRasCommFini(const struct ncclComm* comm);
ncclResult_t ncclRasAddRanks(struct rasRankInit* ranks, int nranks);
ncclRasDiagMode ncclRasDiagGetMode(const struct ncclComm* comm);
ncclResult_t ncclRasPassiveDiagTrigger(struct ncclComm* comm);

#endif // !NCCL_RAS_H_
