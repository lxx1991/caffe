#include <vector>

#include "caffe/layers/feature_selection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeatureSelectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void FeatureSelectionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (head_.count() < bottom[0]->height())
    head_.Reshape(1, 1, 1, bottom[0]->height());

  if (prev_.count() < bottom[1]->count() )
  {
    prev_.Reshape(1, 1, 1, bottom[1]->count());
    ending_.Reshape(1, 1, 1, bottom[1]->count());
  }
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  top[0]->Reshape(bottom[1]->height(), bottom[0]->channels(), 1, bottom[1]->width());
}

template <typename Dtype>
void FeatureSelectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
/*
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_index = bottom[1]->cpu_data(), *bottom_data = bottom[0]->cpu_data();

  for (int n = 0; n < bottom[0]->num(); ++n)
  {
    for (int c = 0; c < bottom[0]->channels(); ++c)
    {
      for (int h = 0; h < bottom[1]->height(); ++h)
        for (int w = 0; w < bottom[1]->width(); ++w)
        {
          const int index = h * bottom[1]->width() + w;
          top_data[index] = bottom_data[int(bottom_index[index])];
        }
      top_data += top[0]->offset(0, 1);
      bottom_data += bottom[0]->offset(0, 1);
    }
    bottom_index += bottom[1]->offset(1);
  }*/
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void FeatureSelectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0]) {
    return;
  }
 /* const Dtype* top_diff = top[0]->cpu_diff(), *bottom_index = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);


  for (int n = 0; n < bottom[0]->num(); ++n)
  {
    for (int c = 0; c < bottom[0]->channels(); ++c)
    {
      for (int h = 0; h < bottom[1]->height(); ++h)
        for (int w = 0; w < bottom[1]->width(); ++w)
        {
          const int index = h * bottom[1]->width() + w;
          bottom_diff[int(bottom_index[index])] += top_diff[index];
        }
      top_diff += top[0]->offset(0, 1);
      bottom_diff += bottom[0]->offset(0, 1);
    }
    bottom_index += bottom[1]->offset(1);
  }*/
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(FeatureSelectionLayer);
#endif

INSTANTIATE_CLASS(FeatureSelectionLayer);
REGISTER_LAYER_CLASS(FeatureSelection);

}  // namespace caffe
