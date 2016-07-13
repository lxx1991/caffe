#ifndef CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_DATA_LAYER_HPP_


#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {


template <typename Dtype>
class ImageSegDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageSegDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageSegDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MaxTopBlobs() const { return 3; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

 protected:
  
  bool output_data_dim_;

  Blob<Dtype> transformed_label_;
  Blob<Dtype> transformed_cat_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  vector< vector<std::string> > lines_;
  int lines_id_;
};



}  // namespace caffe

#endif  // CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
