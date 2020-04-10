// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace utils {

/**
 * Gets the static random seed value. All calls to GetStaticRandomSeed()
 * throughout the lifetime of the process will return the same value.
 *
 * @return The static random seed value.
 */
int64_t GetStaticRandomSeed();

/**
 * Sets the static random seed value.
 *
 * If called, this should be called before calling GetStaticRandomSeed().
 * Not calling this is also fine. In that case, the value will be generated.
 *
 * @param seed The random seed value to use.
 */
void SetStaticRandomSeed(int64_t seed);

}  // namespace test
}  // namespace onnxruntime
