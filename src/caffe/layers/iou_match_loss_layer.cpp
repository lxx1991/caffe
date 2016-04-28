#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/iou_match_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/iou.hpp"
#include "caffe/util/kuhn_munkres.hpp"

namespace caffe {

template <typename Dtype>
void IouMatchLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  lambda_ = this->layer_param_.iou_match_loss_param().lambda();
  way_ = new Blob<Dtype>(1, 1, 1, 1);
  match_ = new Blob<int>(1, 1, 1, 1);
}

template <typename Dtype>
void IouMatchLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::Reshape(bottom, top);
  
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());

  n1_ = bottom[0]->channels();
  n2_ = bottom[1]->channels();
  CHECK_EQ(n2_, bottom[2]->channels());

  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  way_->Reshape(n1_, n2_, 1, 1);
  match_->Reshape(n2_, 1, 1, 1);
}



template <typename Dtype>
void IouMatchLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* label = bottom[0]->cpu_data();
  const Dtype* predict = bottom[1]->cpu_data();
  const Dtype* score = bottom[2]->cpu_data();

  const int len = bottom[1]->height() * bottom[1]->width();
  
  for (int i=0; i<n1_; i++)
    for (int j=0; j<n2_; j++)
      way_->mutable_cpu_data()[way_->offset(i, j)] = iou(label + i * len, predict + j * len, len);

  kuhn_munkres<Dtype>(way_, match_);

  Dtype loss = 0;
  for (int i=0; i<n2_; i++)
    if (match_->cpu_data()[i] >= 0)
      loss -= way_->cpu_data()[way_->offset(match_->cpu_data()[i], i)] + lambda_ * (log(score[i]));
    else
      loss -= lambda_ * (log(1 - score[i]));
  top[0]->mutable_cpu_data()[0] = loss;
}



template <typename Dtype>
void IouMatchLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* label = bottom[0]->cpu_data();
  const Dtype* predict = bottom[1]->cpu_data();
  const Dtype* score = bottom[2]->cpu_data();

  const Dtype clip = this->layer_param_.iou_match_loss_param().clip();

  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[1]) {
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
    const int len = bottom[1]->height() * bottom[1]->width();
    for (int i=0; i<n2_; i++)
    {
      if (match_->cpu_data()[i] >= 0)
        iou_diff(label + match_->cpu_data()[i] * len, predict + i * len, bottom_diff, len);
      else
        caffe_set(len, Dtype(0), bottom_diff);
      if (clip > Dtype(0))
      {
        for (int j=0; j<len; j++)
        {
          if (bottom_diff[j] > clip) bottom_diff[j] = clip;
          if (bottom_diff[j] < -clip) bottom_diff[j] = -clip;
        }
      }
      bottom_diff += len;
    }
  }
  if (propagate_down[2]) {
    Dtype* bottom_diff = bottom[2]->mutable_cpu_diff();
    for (int i=0; i<n2_; i++)
    {
      if (match_->cpu_data()[i] >= 0)
        bottom_diff[i] = -1/score[i];
      else
        bottom_diff[i] = -1/(1-score[i]);
      
      if (clip > Dtype(0))
      {
        if (bottom_diff[i] > clip) bottom_diff[i] = clip;
        if (bottom_diff[i] < -clip) bottom_diff[i] = -clip;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(IouMatchLossLayer);
#endif

INSTANTIATE_CLASS(IouMatchLossLayer);
REGISTER_LAYER_CLASS(IouMatchLoss);

}  // namespace caffe
