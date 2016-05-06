#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/superpixel_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->num(), 1);
	CHECK_EQ(bottom[1]->num(), 1);
	CHECK_EQ(bottom[2]->num(), 1);
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[2]->channels(), 1);
	
	CHECK_EQ(bottom[0]->height(), bottom[1]->height());
	CHECK_EQ(bottom[0]->width(), bottom[1]->width());
	CHECK_EQ(bottom[2]->width(), 1);

	superpixel_num_ = bottom[2]->height();

	top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), superpixel_num_, 1);
	if (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX || this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_AVE)
	{
		max_idx_.Reshape(bottom[0]->num(), bottom[0]->channels(), superpixel_num_, 1);
	}
}


template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data(), *bottom_superpixel = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  const int top_count = top[0]->count();

  int* mask = NULL;  // suppress warnings about uninitalized variables

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
	mask = max_idx_.mutable_cpu_data();
	caffe_set(top_count, -1, mask);
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {

      for (int h = 0; h < bottom[0]->height(); ++h) {
        for (int w = 0; w < bottom[0]->width(); ++w) {
        	const int index = h * bottom[0]->width() + w;
        	int superpixel_id = int(bottom_superpixel[index]);
        	CHECK(superpixel_id < superpixel_num_);
    		  if (bottom_data[index] > top_data[superpixel_id])
      		{	
  				  top_data[superpixel_id] = bottom_data[index];
  			   mask[superpixel_id] = index;
      		}
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      mask += top[0]->offset(0, 1);
    }
  }
  break;
  case PoolingParameter_PoolMethod_AVE:
  	// Initialize
	mask = max_idx_.mutable_cpu_data();
	caffe_set(top_count, 0, mask);
  caffe_set(top_count, Dtype(0), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < bottom[0]->channels(); ++c) {

        for (int h = 0; h < bottom[0]->height(); ++h) {
          for (int w = 0; w < bottom[0]->width(); ++w) {
          	const int index = h * bottom[0]->width() + w;
          	int superpixel_id = int(bottom_superpixel[index]);
          	CHECK(superpixel_id < superpixel_num_);
            top_data[superpixel_id] += bottom_data[index];
		        mask[superpixel_id] += 1;
		  }
		}
        for (int h = 0; h < top[0]->height(); ++h)
        	top_data[h] /= Dtype(mask[h]);
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        mask += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff(), *bottom_superpixel = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  const int* mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
  	mask = max_idx_.cpu_data();

    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < top[0]->channels(); ++c) {
        for (int ph = 0; ph < top[0]->height(); ++ph) {
          for (int pw = 0; pw < top[0]->width(); ++pw) {
            const int index = ph * top[0]->height() + pw;
            const int bottom_index = mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        mask += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
	// The main loop
	mask = max_idx_.cpu_data();
    
  	for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < bottom[0]->channels(); ++c) {
        for (int h = 0; h < bottom[0]->height(); ++h) {
          for (int w = 0; w < bottom[0]->width(); ++w) {
          	const int index = h * bottom[0]->width() + w;
          	int superpixel_id = int(bottom_superpixel[index]);
          	bottom_diff[index] += top_diff[superpixel_id] / Dtype(mask[superpixel_id]);
		  }
		}
        // compute offset
		bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        mask += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(SuperpixelPoolingLayer);
#endif

INSTANTIATE_CLASS(SuperpixelPoolingLayer);
REGISTER_LAYER_CLASS(SuperpixelPooling);

}  // namespace caffe