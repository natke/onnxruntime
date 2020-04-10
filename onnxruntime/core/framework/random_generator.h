// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <stdint.h>
#include <utility>

namespace onnxruntime {

class PhiloxGenerator {
public:
  PhiloxGenerator() : seed_(0), offset_(0) {}
  PhiloxGenerator(uint64_t seed) : seed_(seed), offset_(0) {}

  void SetSeed(uint64_t seed) {
    seed_ = seed;
    offset_.store(0);
  }

  std::pair<uint64_t, uint64_t> GetPhiloxSeeds(uint64_t count) {
    uint64_t offset = offset_.fetch_add(count);
    return std::pair<uint64_t, uint64_t>(seed_, offset);
  }

  static PhiloxGenerator& Default();

 private:
  uint64_t seed_;
  std::atomic<uint64_t> offset_;
};

}  // namespace onnxruntime
