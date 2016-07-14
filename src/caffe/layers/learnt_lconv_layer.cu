#include <vector>

#include "caffe/layers/learnt_lconv_layer.hpp"
#include "caffe/util/math_functions.hpp"



namespace caffe {

template <typename Dtype>
__global__ void LearntLConvForward(const int nthreads,
          const Dtype* bottom_data, const int num, const int channels,
		  const int height, const int width, const Dtype* filter, const int filter_height,
    	  const int filter_width, Dtype* top_data) 
{
	  CUDA_KERNEL_LOOP(index, nthreads) {
	  		int w = index % width;
	  		int h = (index / width) % height;
	  		int c = (index / width / height) % channels;
	  		int n = index / width / height / channels;
	  		for (int i=0; i<filter_height; i++)
	  		{
	  			int n_h = h + i  - (filter_height / 2);
	  			if (n_h >= 0 and n_h < height)
		  			for (int j=0; j<filter_width; j++)
		  			{
		  				int n_w = w + j  - (filter_width / 2);
		  				if (n_w >= 0 and n_w < width)
		  					top_data[index] += bottom_data[((n * channels  + c) * height + n_h)*width + n_w] * filter[((n * filter_height * filter_width  + i*filter_width + j) * height + h)*width + w];
	  				}
  			}
	  }
}


template <typename Dtype>
__global__ void LearntLConvBackward(const int nthreads,
          Dtype* bottom_diff, const Dtype* bottom_data, const int num, const int channels,
		  const int height, const int width, Dtype* filter_diff, const Dtype* filter, const int filter_height,
    	  const int filter_width, const Dtype* top_diff) 
{
	  CUDA_KERNEL_LOOP(index, nthreads) {
	  		int w = index % width;
	  		int h = (index / width) % height;
	  		int c = (index / width / height) % channels;
	  		int n = index / width / height / channels;
	  		for (int i=0; i<filter_height; i++)
	  		{
	  			int n_h = h + i  - (filter_height / 2);
	  			if (n_h >= 0 and n_h < height)
		  			for (int j=0; j<filter_width; j++)
		  			{
		  				int n_w = w + j  - (filter_width / 2);
		  				if (n_w >= 0 and n_w < width)
		  				{
		  					bottom_diff[((n * channels  + c) * height + n_h)*width + n_w] += top_diff[index] * filter[((n * filter_height * filter_width  + i*filter_width + j) * height + h)*width + w];
		  					filter_diff[((n * filter_height * filter_width  + i*filter_width + j) * height + h)*width + w] += top_diff[index] * bottom_data[((n * channels  + c) * height + n_h)*width + n_w];
  						}
  					}
			}
	  }
}

template <typename Dtype>
void LearntLConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_filter = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  caffe_gpu_set(count, Dtype(0.), top_data);

  int filter_height = int(std::sqrt(bottom[1]->channels()));
  int filter_width = filter_height;
  CHECK_EQ(filter_height * filter_width, bottom[1]->channels());

  LearntLConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top[0]->num(), top[0]->channels(), top[0]->height(), top[0]->width(), bottom_filter, filter_height, filter_width, top_data);
  
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void LearntLConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count = top[0]->count();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_gpu_diff());
  caffe_gpu_set(bottom[1]->count(), Dtype(0.), bottom[1]->mutable_gpu_diff());

  int filter_height = int(std::sqrt(bottom[1]->channels()));
  int filter_width = filter_height;
  CHECK_EQ(filter_height * filter_width, bottom[1]->channels());

  LearntLConvBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[0]->mutable_gpu_diff(), bottom[0]->gpu_data(), top[0]->num(), top[0]->channels(), top[0]->height(), top[0]->width(), bottom[1]->mutable_gpu_diff(), bottom[1]->gpu_data(), filter_height, filter_width, top[0]->gpu_diff());
  
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(LearntLConvLayer);


}  // namespace caffe