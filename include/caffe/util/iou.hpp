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
		y = y - x;
		CHECK_GE(y, x);

		if (y < 1e-8)
			return 1;
		else
			return x / y;
	}

	template <typename Dtype>
	void iou_diff(const Dtype* label, const Dtype* predict, Dtype* diff, const int len)
	{
		Dtype x = Dtype(0), y = Dtype(0), a, b;
		for (int i=0; i<len; i++)
		{
			x += label[i] * predict[i];
			y += label[i] + predict[i];
		}
		y = y - x;
		CHECK_GE(y, x);
		
		if (y < 1e-8)
		{
			a = Dtype(1) / 1e-8;
			b = Dtype(-1) / 1e-8;
		}
		else
		{
			a = Dtype(1) / y;
			b = Dtype(-x) / (y * y);
		}
		for (int i=0; i<len; i++)
			diff[i] = label[i] * a + (Dtype(1) - label[i]) * b;
	}

}

#endif 	//CAFFE_IOU_HPP_