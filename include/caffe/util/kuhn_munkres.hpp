#ifndef CAFFE_KUHN_MUNKRES_HPP_
#define CAFFE_KUHN_MUNKRES_HPP_

#include <algorithm>
#include <cfloat>

#include "caffe/blob.hpp"

namespace caffe {

	template <typename Dtype>
	bool find(int i, int n, Dtype *edge, Dtype *lx, Dtype *ly, int *my, Dtype *slack, bool *vx, bool *vy)
	{
		vx[i] = true;
		for (int j=0; j<n; j++)
			if (!vy[j])
			{
				Dtype t = lx[i] + ly[j] - edge[i*n + j];
				if (std::abs(t) < 1e-10)
				{
					vy[j] = true;
					if (my[j] == -1 || find(my[j], n, edge, lx, ly, my, slack, vx, vy))
					{
						my[j] = i;
						return true;
					}
				}
				else if (slack[j] > t)
					slack[j] = t;
			}
		return false;
	}

	template <typename Dtype>
	void kuhn_munkres(Blob<Dtype> *way, Blob<int> *match)
	{
		int n = std::max(way->num(), way->channels());
		Dtype *edge, *lx, *ly, *slack;
		int *my;
		bool *vx, *vy;
		edge = new Dtype[n*n];
		lx = new Dtype[n];
		ly = new Dtype[n];
		my = new int[n];
		slack = new Dtype[n];
		vx = new bool[n];
		vy = new bool[n];
		memset(lx, 0, sizeof(Dtype) * n);
		memset(ly, 0, sizeof(Dtype) * n);
		memset(my, 255, sizeof(int) * n);
		for (int i = 0; i<n; i++)
			for (int j=0; j<n; j++)
			{
				if (i<way->num() && j < way->channels())
				{
					DCHECK_GE(way->cpu_data()[way->offset(i, j)], 0);
					edge[i*n+j] = way->cpu_data()[way->offset(i, j)];
				}
				else
					edge[i*n+j] = 0;
				lx[i] = std::max(lx[i], edge[i*n+j]);
			}
		for (int i=0; i<n; i++)
		{
			for (int j=0; j<n; j++)
				slack[j] = FLT_MAX;
			while (true)
			{
				memset(vx, 0, sizeof(bool) * n);
				memset(vy, 0, sizeof(bool) * n);
				if (find(i, n, edge, lx, ly, my, slack, vx, vy))
					break;
				Dtype d = FLT_MAX;
				for (int j=0; j<n; j++)
					if (!vy[j] && slack[j] < d)
						d = slack[j];
				for (int j=0; j<n; j++)
				{
					if (vx[j])
						lx[j] -= d;
					if (vy[j])
						ly[j] += d;
				}
			}
		}
		for (int i=0; i<match->num(); i++)
		{
			if (my[i] < way->num())
				match->mutable_cpu_data()[i] = my[i];
			else
				match->mutable_cpu_data()[i] = -1;
			//LOG(ERROR) << match->mutable_cpu_data()[i];
		}
		delete []edge;
		delete []lx;
		delete []ly;
		delete []my;
		delete []slack;
		delete []vx;
		delete []vy;
	}


}

#endif 	//CAFFE_KUHN_MUNKRES_HPP_