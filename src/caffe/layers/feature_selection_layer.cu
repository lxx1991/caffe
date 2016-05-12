#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/feature_selection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FeatureSelectionForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const bottom_index, const int channels,
    const int height, const int height_index, const int width_index, 
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width_index;
    const int c = (index / width_index) % channels;
    const int n = index / width_index / channels;
    const int n_idx = int(bottom_index[n * width_index + w]);
    top_data[index] = bottom_data[c *  height + n_idx];
  }
}



template <typename Dtype>
void FeatureSelectionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_index = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  FeatureSelectionForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_index, bottom[0]->channels(),
        bottom[0]->height(), bottom[1]->height(), bottom[1]->width(), top_data);

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void FeatureSelectionBackward(const int nthreads, const Dtype* const top_diff, const int channels,
    const int height, const int width,
    const int* const ending, const int* const prev, const int* const head,
    Dtype* const bottom_diff) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int h = index % height;
    const int c = index / height;
    for (int n = head[h]; n!=-1; n = prev[n])
      bottom_diff[index] += top_diff[ (ending[n] / width) * channels * width + c * width + (ending[n] % width) ];
  }
}

template <typename Dtype>
void FeatureSelectionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_index = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int *head = head_.mutable_cpu_data(), *prev = prev_.mutable_cpu_data(), *ending = ending_.mutable_cpu_data();
  caffe_set(bottom[0]->height(), -1, head);
  int index = 0;
  for (int h=0; h<bottom[1]->height(); h++)
    for (int w=0; w<bottom[1]->width(); w++)
    {
      int x = int(bottom_index[index]), y = h * bottom[1]->width() + w;
      ending[index] = y;
      prev[index] = head[x];
      head[x] = index;
      index = index + 1;
    }
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  //LOG(ERROR) << "start";
  FeatureSelectionBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, bottom[0]->channels(), bottom[0]->height(), bottom[1]->width(), 
        ending_.gpu_data(), prev_.gpu_data(), head_.gpu_data(), bottom_diff);
  //LOG(ERROR) << "end";
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(FeatureSelectionLayer);


}  // namespace caffe
