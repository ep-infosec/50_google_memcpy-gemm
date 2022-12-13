// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/matrix_lib.h"

#include "src/matrix_lib_impl.h"

namespace {

using ::half_float::half;
using ::matrix_lib::internal::FillGaussian;
using ::matrix_lib::internal::FillUniform;
using ::matrix_lib::internal::kBaseMatrixSize;
}

// Do base fill in float and then convert to half precision.
template <>
bool matrix_lib::internal::FillArray<half>(half *A, int n, absl::BitGen *rng,
                                           float scale, bool nv_gauss) {
  auto baseMatrix = std::make_unique<float[]>(kBaseMatrixSize);
  if (nv_gauss) {
    FillGaussian<float>(absl::Span<float>(baseMatrix.get(), kBaseMatrixSize),
                        rng);
  } else {
    FillUniform<float>(absl::Span<float>(baseMatrix.get(), kBaseMatrixSize),
                       rng, scale);
  }
  for (int i = 0; i < n; i++) {
    A[i] = half(baseMatrix[i % kBaseMatrixSize]);
  }
  return true;
}

template class RandomMatrix<int8_t>;
template class RandomMatrix<float>;
template class RandomMatrix<double>;
template class RandomMatrix<half>;
