#include <cfloat>
#include <vector>

#include "caffe/layers/learnt_lconv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LearntLConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void LearntLConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  	CHECK(bottom[1]->num() == bottom[0]->num());
	CHECK(bottom[1]->height() == bottom[0]->height());
	CHECK(bottom[1]->width() == bottom[0]->width());
  	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void LearntLConvLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void LearntLConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(LearntLConvLayer);
#endif

INSTANTIATE_CLASS(LearntLConvLayer);
REGISTER_LAYER_CLASS(LearntLConv);

}  // namespace caffe
