/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/kernels/internal/mfcc.h"

#include <stddef.h>
#include <stdint.h>

#include <vector>

// #include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace resampler {

enum KernelType {
  kReference,
};

// typedef struct {
//   float upper_frequency_limit;
//   float lower_frequency_limit;
//   int filterbank_channel_count;
//   int dct_coefficient_count;
// } TfLiteMfccParams;

constexpr int kInputTensorWav = 0;
constexpr int kInputTensorRate = 1;
constexpr int kOutputTensor = 0;


TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* feat_map;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorWav, &feat_map));
  const TfLiteTensor* sample_pt;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorRate, &sample_pt));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_EQ(context, NumDimensions(feat_map), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(sample_pt), 4);

  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, feat_map->type, output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, sample_pt->type, kTfLiteFloat32);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = feat_map->dims->data[0];
  output_size->data[1] = feat_map->dims->data[1];
  output_size->data[2] = feat_map->dims->data[2];
  output_size->data[3] = feat_map->dims->data[3];

  return context->ResizeTensor(context, output, output_size);
}

// Input is a single squared-magnitude spectrogram frame. The input spectrum
// is converted to linear magnitude and weighted into bands using a
// triangular mel filterbank, and a discrete cosine transform (DCT) of the
// values is taken. Output is populated with the lowest dct_coefficient_count
// of these values.
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_wav;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorWav, &input_wav));
  const TfLiteTensor* input_rate;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorRate, &input_rate));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // const int spectrogram_channels = input_wav->dims->data[2];
  // const int spectrogram_samples = input_wav->dims->data[1];
  // const int audio_channels = input_wav->dims->data[0];

  // const float* spectrogram_flat = GetTensorData<float>(input_wav);
  // float* output_flat = GetTensorData<float>(output);
  return kTfLiteOk;
}

}  // namespace mfcc

TfLiteRegistration* Register_RESAMPLER() {
  static TfLiteRegistration r = {nullptr, nullptr, resampler::Prepare, resampler::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
