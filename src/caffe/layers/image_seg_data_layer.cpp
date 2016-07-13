#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/image_seg_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
ImageSegDataLayer<Dtype>::~ImageSegDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const int label_type = this->layer_param_.image_data_param().label_type();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  TransformationParameter transform_param = this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false) << 
         "ImageSegDataLayer does not support mean file";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  string linestr;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    string segfn = "";
    if (label_type != ImageDataParameter_LabelType_NONE) {
      iss >> segfn;
    }
    if (label_type == ImageDataParameter_LabelType_BBOX || label_type == ImageDataParameter_LabelType_PER_CAT_BBOX) {
      string bboxfn;
      iss >> bboxfn;
      bboxfn = root_folder + bboxfn;
      FILE* bboxfile = fopen(bboxfn.c_str(), "r");
      int nbox;
      CHECK_EQ(fscanf(bboxfile, "%d", &nbox), 1);
      for (int i = 0; i < nbox; i++)
      {
        vector<std::string> file_list;
        int k;
        char c[50];
        file_list.push_back(imgfn);
        file_list.push_back(segfn);
        for (int j = 0; j < 5; j++)
        {
          CHECK_EQ(fscanf(bboxfile, "%d", &k), 1);
          sprintf(c, "%d", k);
          file_list.push_back(std::string(c));
        }
        if (fgets(c, sizeof(c), bboxfile) != NULL)
        {
          double p = 1.0;
          if (strlen(c) > 1)
            sscanf(c, "%lf\n", &p);
          //if (p > 0.8)
          lines_.push_back(file_list);
        }
        else
          lines_.push_back(file_list);
      }
      fclose(bboxfile);
    }
    else
    {
      vector<std::string> file_list;
      file_list.push_back(imgfn);
      file_list.push_back(segfn);
      lines_.push_back(file_list);
    }
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
                                    new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  int crop_size_h = this->layer_param_.transform_param().crop_size_h();
  int crop_size_w = this->layer_param_.transform_param().crop_size_w();
  
  if (crop_size_h == 0)
    crop_size_h = crop_size;
  if (crop_size_w == 0)
    crop_size_w = crop_size;

  const int batch_size = this->layer_param_.image_data_param().batch_size();
  const int label_crop_size = this->layer_param_.transform_param().label_crop_size();
  if (crop_size_h > 0) {
    top[0]->Reshape(batch_size, channels, crop_size_h, crop_size_w);

    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, channels, crop_size_h, crop_size_w);
    }
    this->transformed_data_.Reshape(1, channels, crop_size_h, crop_size_w);

    //label
    top[1]->Reshape(batch_size, 1, crop_size_h - 2 * label_crop_size, crop_size_w - 2 * label_crop_size);

    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 1, crop_size_h - 2 * label_crop_size, crop_size_w - 2 * label_crop_size);
    }
    this->transformed_label_.Reshape(1, 1, crop_size_h - 2 * label_crop_size, crop_size_w - 2 * label_crop_size);

  } else {
    top[0]->Reshape(batch_size, channels, height, width);

    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, channels, height, width);
    }
    this->transformed_data_.Reshape(1, channels, height, width);

    //label
    top[1]->Reshape(batch_size, 1, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 1, height, width);
    }
    this->transformed_label_.Reshape(1, 1, height, width);     
  }
  // Per_Category_Label
  if (label_type == ImageDataParameter_LabelType_PER_CAT_BBOX)
  {
    top[2]->Reshape(batch_size, 1, 1, 1);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].attach_.Reshape(batch_size, 1, 1, 1);
    }
    this->transformed_cat_.Reshape(1, 1, 1, 1);
  }
  
    

  LOG(INFO) << "output data size: " << top[0]->num() << ","
	    << top[0]->channels() << "," << top[0]->height() << ","
	    << top[0]->width();
  // label
  LOG(INFO) << "output label size: " << top[1]->num() << ","
	    << top[1]->channels() << "," << top[1]->height() << ","
	    << top[1]->width();
  // image_dim
  if (label_type == ImageDataParameter_LabelType_PER_CAT_BBOX)
  {
    LOG(INFO) << "output data_dim size: " << top[2]->num() << ","
  	    << top[2]->channels() << "," << top[2]->height() << ","
  	    << top[2]->width();
  }
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageSegDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  Dtype* top_data     = batch->data_.mutable_cpu_data();
  Dtype* top_label    = batch->label_.mutable_cpu_data();
  Dtype* top_cat = NULL;
  ImageDataParameter image_data_param    = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const int label_type = this->layer_param_.image_data_param().label_type();
  const int ignore_label = image_data_param.ignore_label();
  const bool is_color  = image_data_param.is_color();
  string root_folder   = image_data_param.root_folder();

  if (label_type == ImageDataParameter_LabelType_PER_CAT_BBOX)
  {
    top_cat = batch->attach_.mutable_cpu_data();
  }

  const int lines_size = lines_.size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {

    std::vector<cv::Mat> cv_img_seg;

    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

    
    cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_][0],
	  new_height, new_width, is_color));

    if (!cv_img_seg[0].data) {
      DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_][0];
    }
    if (label_type == ImageDataParameter_LabelType_PIXEL || label_type == ImageDataParameter_LabelType_BBOX || ImageDataParameter_LabelType_PER_CAT_BBOX) {
      cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_][1],
					    new_height, new_width, false));
      if (!cv_img_seg[1].data) {
      DLOG(INFO) << "Fail to load seg: " << root_folder + lines_[lines_id_][1];
      }
    }
    else if (label_type == ImageDataParameter_LabelType_IMAGE) {
      const int label = atoi(lines_[lines_id_][1].c_str());
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
		  CV_8UC1, cv::Scalar(label));
      cv_img_seg.push_back(seg);      
    }
    else {
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
		  CV_8UC1, cv::Scalar(ignore_label));
      cv_img_seg.push_back(seg);
    }

    if (this->layer_param_.transform_param().crop_resize_size() > 0)
    {
      CHECK_EQ(this->layer_param_.transform_param().crop_resize_size(), 2);
      const float crop_resize0 = this->layer_param_.transform_param().crop_resize(0);
      const float crop_resize1 = this->layer_param_.transform_param().crop_resize(1);
      const float temp_scale = crop_resize0 + (crop_resize1 - crop_resize0) * (caffe_rng_rand() % 101l) / 100.0f;
      resize(cv_img_seg[0], cv_img_seg[0], cv::Size(0, 0), temp_scale, temp_scale);
      resize(cv_img_seg[1], cv_img_seg[1], cv::Size(0, 0), temp_scale, temp_scale, CV_INTER_NN);
    }

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset, ret = 0;

    offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);

    offset = batch->label_.offset(item_id);
    this->transformed_label_.set_cpu_data(top_label + offset);

    if (label_type == ImageDataParameter_LabelType_PER_CAT_BBOX)
    {
      offset = batch->attach_.offset(item_id);
      this->transformed_cat_.set_cpu_data(top_cat + offset);
    }

    if (label_type == ImageDataParameter_LabelType_BBOX || label_type == ImageDataParameter_LabelType_PER_CAT_BBOX)
    {
      cv::Rect Bbox = cv::Rect( atoi(lines_[lines_id_][2].c_str()), 
                                atoi(lines_[lines_id_][3].c_str()), 
                                atoi(lines_[lines_id_][4].c_str()) - 1, 
                                atoi(lines_[lines_id_][5].c_str()) - 1);
      int label_num = atoi(lines_[lines_id_][6].c_str());
      /*LOG(ERROR) << "----------------------------------------------------------------------";
      LOG(ERROR) << lines_[lines_id_][2] << ' ' <<lines_[lines_id_][3] << ' '<<lines_[lines_id_][4] << ' '<<lines_[lines_id_][5]<< ' '<<lines_[lines_id_][6];
      LOG(ERROR) << cv_img_seg[0].cols << ' ' <<cv_img_seg[0].rows;*/
      ret = this->data_transformer_->TransformImgAndSeg_Bbox(cv_img_seg, &(this->transformed_data_), &(this->transformed_label_), ignore_label, Bbox, label_num, label_type == ImageDataParameter_LabelType_PER_CAT_BBOX);
      
      if (label_type == ImageDataParameter_LabelType_PER_CAT_BBOX)
        this->transformed_cat_.mutable_cpu_data()[0] = label_num;
    }
    else
      this->data_transformer_->TransformImgAndSeg(cv_img_seg, &(this->transformed_data_), &(this->transformed_label_), ignore_label);
    
    // save top
    /*
  	if (ret != -1)
  	{
	  	cv::Mat im_data(this->transformed_data_.height(), this->transformed_data_.width(), CV_8UC3);
	  	cv::Mat im_label(this->transformed_label_.height(), this->transformed_label_.width(), CV_8UC1);

	  	for (int p1 = 0; p1 < this->transformed_data_.height(); p1 ++)
	  		for (int p2 = 0; p2 < this->transformed_data_.width(); p2 ++)
	  		{
	  			im_data.at<uchar>(p1, p2*3+0) = (uchar)(this->transformed_data_.data_at(0, 0, p1, p2)+104.00698793);
	  			im_data.at<uchar>(p1, p2*3+1) = (uchar)(this->transformed_data_.data_at(0, 1, p1, p2)+116.66876762);
	  			im_data.at<uchar>(p1, p2*3+2) = (uchar)(this->transformed_data_.data_at(0, 2, p1, p2)+122.67891434);	
	  		}
  		for (int p1 = 0; p1 < this->transformed_label_.height(); p1 ++)
  	  		for (int p2 = 0; p2 < this->transformed_label_.width(); p2 ++)
  	  			im_label.at<uchar>(p1, p2) = this->transformed_label_.data_at(0, 0, p1, p2);
	  	static int tot = 0;
	  	tot = tot + 1;
	  	char temp_path[100];
	  	sprintf(temp_path, "temp/%d_0.jpg", tot);
	  	imwrite(temp_path, cv_img_seg[0]);
	  	sprintf(temp_path, "temp/%d_1.jpg", tot);
	  	imwrite(temp_path, cv_img_seg[1]);
	  	sprintf(temp_path, "temp/%d_1.jpg", tot);
	  	imwrite(temp_path, im_data);
		  sprintf(temp_path, "temp/%d_2.png", tot);
	  	imwrite(temp_path, im_label);
  	}
    */
    trans_time += timer.MicroSeconds();

    // go to the next std::vector<int>::iterator iter;
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
	       ShuffleImages();
      }
    }
    item_id += ret;
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSegDataLayer);
REGISTER_LAYER_CLASS(ImageSegData);
}  // namespace caffe
