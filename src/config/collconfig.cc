/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#include "config/collconfig.h"
#include "core.h" // WARN
#include "device.h" // MAXCHANNELS
#include "nccl.h"

// Helper function to set aggregation isolation
bool ncclCollConfigNeedAggIsolate(const ncclCollConfig_t* config) {
  if (config->size == 0) {
    // zero size config indicate no user config is passed, no aggregation isolation
    return false;
  }
  // check for other options to determine if aggregation isolation is needed
  if (config->minCTAs != NCCL_CONFIG_UNDEF_INT || config->maxCTAs != NCCL_CONFIG_UNDEF_INT) return true;
  if (config->nvlsCTAs != NCCL_CONFIG_UNDEF_INT) return true;
  // cgaClusterSize is a per-launch attribute applied to the whole plan, not a per-collective resource
  // cap, so it does not force aggregation isolation; it is applied to the group as-is.
  return false;
}

// Validate a per-call config header and copy it into internal_config for safe access during
// scheduling. A NULL user config is accepted (no config); internal_config->size is set to 0 when no
// user config was passed (used to isolate configured collectives from aggregation), non-zero
// otherwise. Later commits add per-call resource knobs, resolved (env > per-call > comm) by their
// consumers via resolveCollRes.
ncclResult_t ncclParseCollConfig(const ncclCollConfig_t* config, ncclCollConfig_t* internal_config) {
  *internal_config = NCCL_COLLCONFIG_INITIALIZER;
  if (config != nullptr) {
    // ncclCollConfig_t is append-only (size only grows, fields never move). The smallest valid
    // struct is the v23100 one, used as the minimal size check.
    if (config->size < sizeof(struct ncclCollConfig_v23100)) {
      WARN("ncclCollConfig_t size %zu too small (expected >= %zu)", config->size, sizeof(ncclCollConfig_t));
      return ncclInvalidArgument;
    }
    if (config->magic != NCCL_API_MAGIC) {
      WARN("ncclCollConfig_t magic mismatch (got 0x%x, expected 0x%x)", config->magic, NCCL_API_MAGIC);
      return ncclInvalidArgument;
    }
    // copy user config to internal_config for safe access later
    if (config->version <= NCCL_VERSION_CODE) {
      // A same/older user config struct is smaller than (or equal to) current library's struct.
      memcpy(internal_config, config, config->size);
    } else {
      // A newer user config struct is larger than current library's struct
      memcpy(internal_config, config, internal_config->size);
    }
  } else {
    // When no user config, set collConfig.size == 0 to mark as "no user config passed"
    internal_config->size = 0;
  }
  return ncclSuccess;
}
