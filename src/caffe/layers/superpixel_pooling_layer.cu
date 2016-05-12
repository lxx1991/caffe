#include <algorithm>
#include <cfloat>
#include <vector>
#include <map>


#include "caffe/layers/superpixel_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"


namespace caffe {

using std::map;

template <typename Dtype>
__global__ void AveSuperpixelPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int channels,
    const int height, const int width, const int superpixel_num_,
    const int* head, const int* ending, const Dtype* weight, const int* prev,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int h_t = index % superpixel_num_;
    const int c = (index / superpixel_num_) % channels;
    const int base_index = c * height * width;
    for (int h=head[h_t]; h!=-1; h = prev[h])
      top_data[index] += bottom_data[base_index + ending[h]] * weight[h];
  }
}


template <typename Dtype>
void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    Dtype* data = blob->mutable_cpu_data();
    int f = ceil(blob->width() / 2.);
    float c = (2 * f - 1 - f % 2) / (2. * f);
    for (int i = 0; i < blob->count(); ++i) {
      float x = i % blob->width();
      float y = (i / blob->width()) % blob->height();
      data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
    }
 }


template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  if (this->layer_param_.pooling_param().pool() != PoolingParameter_PoolMethod_AVE)
  {
    NOT_IMPLEMENTED;
    return;
  }

  const Dtype* bottom_sp = bottom[1]->cpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  int count = top[0]->count();

  int kernel_size = 2 * factor_ - factor_ % 2, pad = ceil((factor_ - 1) / 2.);
  Blob<Dtype> filter(1, 1, kernel_size, kernel_size);
  Fill(&filter);
  int h = bottom[0]->height(), w = bottom[0]->width(), sh = bottom[1]->height(), sw = bottom[1]->width();

  int *ending = ending_.mutable_cpu_data(), *prev = prev_.mutable_cpu_data(), *head = head_.mutable_cpu_data();
  int *ending_bp = ending_bp_.mutable_cpu_data(), *prev_bp = prev_bp_.mutable_cpu_data(), *head_bp = head_bp_.mutable_cpu_data();
  Dtype *weight = weight_.mutable_cpu_data(), *weight_bp = weight_bp_.mutable_cpu_data(); 

  caffe_set(superpixel_num_, int(-1), head);
  caffe_set(h * w, int(-1), head_bp);

  vector<float> pixel_tot(superpixel_num_, 0.);
  for (int i=0; i<sh * sw; i++)
    pixel_tot[bottom_sp[i]]++;

  int edge_tot = 0;
  map<int, float> hash;
  for (int i=0; i<h; i++)
    for (int j=0; j<w; j++)
    {
      int index = i * w + j;
      const int stx = i*factor_ - pad, sty = j*factor_ - pad;
      hash.clear();
      for (int dx = 0; dx < kernel_size; dx++)
        if (dx + stx >=0 && dx + stx < sh)
          for (int dy = 0; dy < kernel_size; dy++)
            if (dy + sty >=0 && dy + sty < sw)
            {
              const int sindex = (dx + stx) * sw + (dy + sty);
              if (hash.find(bottom_sp[sindex]) == hash.end())
                hash[bottom_sp[sindex]] = float(filter.data_at(0, 0, dx, dy));
              else
                hash[bottom_sp[sindex]] += float(filter.data_at(0, 0, dx, dy));
            }

      for (map<int,float>::iterator iter=hash.begin(); iter!=hash.end(); iter++)
      {
        CHECK(edge_tot < ending_bp_.count());
        CHECK(index < head_bp_.count());
        CHECK(iter->first < head_.count());

        ending_bp[edge_tot] = iter->first;
        weight_bp[edge_tot] = Dtype(iter->second / pixel_tot[iter->first]) ;
        prev_bp[edge_tot] = head_bp[index];
        head_bp[index] = edge_tot;

        ending[edge_tot] = index;
        weight[edge_tot] = Dtype(iter->second / pixel_tot[iter->first]);
        prev[edge_tot] = head[iter->first];
        head[iter->first] = edge_tot;
        edge_tot++;
      }
    }
  caffe_gpu_set(count, Dtype(0.), top_data);
  AveSuperpixelPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->channels(), h, w, superpixel_num_, head_.gpu_data(), ending_.gpu_data(), weight_.gpu_data(), prev_.gpu_data(), top_data);
  
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void AveSuperpixelPoolBackward(const int nthreads,
    const Dtype* const top_diff, const int channels,
    const int height, const int width, const int superpixel_num_,
    const int* head, const int* ending, const Dtype* weight, const int* prev,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int base_index = c * superpixel_num_;
    for (int h_t=head[h * width + w]; h_t!=-1; h_t = prev[h_t])
      bottom_diff[index] += top_diff[base_index + ending[h_t]] * weight[h_t];
  }
}

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = bottom[0]->count();
  int h = bottom[0]->height(), w = bottom[0]->width();

  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  AveSuperpixelPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
     count, top_diff, bottom[0]->channels(), h, w, superpixel_num_, head_bp_.gpu_data(), ending_bp_.gpu_data(), weight_bp_.gpu_data(), prev_bp_.gpu_data(), bottom_diff);

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(SuperpixelPoolingLayer);


}  // namespace caffe
