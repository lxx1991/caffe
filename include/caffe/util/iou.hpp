#ifndef CAFFE_IOU_HPP_
#define CAFFE_IOU_HPP_

#include <algorithm>
#include <cfloat>

namespace caffe {

	template <typename Dtype>
	Dtype iou(const Dtype* label, const Dtype* predict, const int len)
	{
		Dtype x = Dtype(0), y = Dtype(0);
		for (int i=0; i<len; i++)
		{
			x += label[i] * predict[i];
			y += label[i] + predict[i];
		}
		y = std::max(y - x, Dtype(FLT_MIN));
		return x / y;
	}

	template <typename Dtype>
	void iou_diff(const Dtype* label, const Dtype* predict, Dtype* diff, const int len)
	{
		Dtype x = Dtype(0), y = Dtype(0);
		for (int i=0; i<len; i++)
		{
			x += label[i] * predict[i];
			y += label[i] + predict[i];
		}
		Dtype a = Dtype(1) / std::max(y - x, Dtype(FLT_MIN)),  b = Dtype(-y) /  std::max((y - x) * (y - x), Dtype(FLT_MIN));
		for (int i=0; i<len; i++)
		{
			diff[i] = label[i] * a + (Dtype(1) - label[i]) * b;
		}
	}

}

#endif 	//CAFFE_IOU_HPP_