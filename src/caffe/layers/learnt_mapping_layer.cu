#include <vector>

#include "caffe/layers/learnt_mapping_layer.hpp"
#include "caffe/util/math_functions.hpp"



namespace caffe {


template <typename Dtype>
__global__ void LearntMappingForward(const int nthreads,
          const Dtype* bottom_data, const int num, const int channels,
		  const int height, const int width, const Dtype* filter, const int filter_channels, const int filter_height,
    	  const int filter_width, Dtype* top_data) 
{
	  CUDA_KERNEL_LOOP(index, nthreads) {
	  		int w = index % width;
	  		int h = (index / width) % height;
	  		int c = (index / width / height) % channels;
	  		int n = index / width / height / channels;
	  		int filter_th = h * width + w;
	  		for (int i=0; i<filter_height; i++)
	  			for (int j=0; j<filter_width; j++)
	  					top_data[index] += bottom_data[((n * channels  + c) * filter_height + i)*filter_width + j] * filter[((n * filter_channels + filter_th) * filter_height + i)*filter_width + j];
	  }
}


template <typename Dtype>
__global__ void LearntMappingBackward(const int nthreads,
          Dtype* bottom_diff, const Dtype* bottom_data, const int num, const int channels,
		  const int height, const int width, Dtype* filter_diff, const Dtype* filter, const int filter_channels, const int filter_height,
    	  const int filter_width, const Dtype* top_diff) 
{
	  CUDA_KERNEL_LOOP(index, nthreads) {
	  		int w = index % width;
	  		int h = (index / width) % height;
	  		int c = (index / width / height) % channels;
	  		int n = index / width / height / channels;
	  		int filter_th = h * width + w;
	  		for (int i=0; i<filter_height; i++)
				for (int j=0; j<filter_width; j++)
				{
					bottom_diff[((n * channels  + c) * filter_height + i)*filter_width + j] += top_diff[index] * filter[((n * filter_channels + filter_th) * filter_height + i)*filter_width + j];
					filter_diff[((n * filter_channels + filter_th) * filter_height + i)*filter_width + j] += top_diff[index] * bottom_data[((n * channels  + c) * filter_height + i)*filter_width + j];
				}
	  }
}


template <typename Dtype>
void LearntMappingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_filter = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  caffe_gpu_set(count, Dtype(0.), top_data);

  LearntMappingForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top[0]->num(), top[0]->channels(), top[0]->height(), top[0]->width(), bottom_filter, bottom[1]->channels(), bottom[1]->height(), bottom[1]->width(), top_data);
  
  CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
void LearntMappingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count = top[0]->count();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_gpu_diff());
  caffe_gpu_set(bottom[1]->count(), Dtype(0.), bottom[1]->mutable_gpu_diff());

  int filter_height = int(std::sqrt(bottom[1]->channels()));
  int filter_width = filter_height;
  CHECK_EQ(filter_height * filter_width, bottom[1]->channels());

  LearntMappingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[0]->mutable_gpu_diff(), bottom[0]->gpu_data(), top[0]->num(), top[0]->channels(), top[0]->height(), top[0]->width(), bottom[1]->mutable_gpu_diff(), bottom[1]->gpu_data(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width(), top[0]->gpu_diff());
  
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(LearntMappingLayer);


}  // namespace caffe