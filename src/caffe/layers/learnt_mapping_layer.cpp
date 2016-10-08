#include <cfloat>
#include <vector>

#include "caffe/layers/learnt_mapping_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LearntMappingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void LearntMappingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK(bottom[0]->num() == bottom[1]->num());
  CHECK(bottom[0]->height() == bottom[0]->width());
  CHECK(bottom[0]->height() == bottom[1]->height());
  CHECK(bottom[0]->width() == bottom[1]->width());
  CHECK(bottom[0]->height() * bottom[0]->width() == bottom[1]->channels());
  
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void LearntMappingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void LearntMappingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(LearntMappingLayer);
#endif

INSTANTIATE_CLASS(LearntMappingLayer);
REGISTER_LAYER_CLASS(LearntMapping);

}  // namespace caffe
