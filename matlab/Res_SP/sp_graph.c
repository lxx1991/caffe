#include "mex.h"
#include "string.h"
#include "assert.h"

#define MAX_NUM 1000
#define MAX_CLASS 30

int cnt_class[MAX_NUM*MAX_CLASS];
int sp_class[MAX_NUM], sp_index[MAX_NUM];

int tot, num, output_size, num_class, num_instance;

void sp_graph(mxArray *plhs[], double *x, int n, int m, double *label)
{
  tot = 0, num = 0, output_size = 0, num_class = 0, num_instance = 0;
  int i, j, k;
  
  memset(sp_index, 0, sizeof(int) * MAX_NUM);
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
    {
      if (x[i*m+j] > num)
        num = x[i*m+j];
      sp_index[(int)x[i*m+j]] = 1; 
    }  
  sp_index[0]--;
  for (i=1 ; i<=num; i++)
      sp_index[i] = sp_index[i] + sp_index[i-1];
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
        x[i*m+j] = sp_index[(int)x[i*m+j]];
  num = 0;
  
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
    {
      if (label[i*m+j] > num_class && label[i*m+j] != 255)
        num_class = label[i*m+j];
      if (x[i*m+j] > num)
        num = x[i*m+j];
    }
  
  num = num + 1;
  num_class = num_class + 1;
  
  assert(num <= MAX_NUM);
  assert(num_class <= MAX_CLASS);
 
  memset(cnt_class, 0, sizeof(int) * num * num_class);
  memset(sp_class, 0, sizeof(int) * num);

  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
        if (label[i*m+j] != 255)
            cnt_class[(int)(x[i*m+j] * num_class + label[i*m+j])]++;
  
  for (i=0; i<num; i++)
  {
    for (j=1; j<num_class; j++)
      if (cnt_class[i * num_class + j] > cnt_class[i * num_class + sp_class[i]])
        sp_class[i] = j;
    if (cnt_class[i * num_class + sp_class[i]] == 0)
        sp_class[i] = 255;
  }
  
  plhs[0] = mxCreateDoubleMatrix(num, 1, mxREAL);
  double *outMatrix = mxGetPr(plhs[0]);
  for (i=0; i<num; i++)
    outMatrix[i] = sp_class[i];
  
  
  plhs[1] = mxCreateDoubleMatrix(m, n, mxREAL);
  outMatrix = mxGetPr(plhs[1]);
  for (i=0; i<n*m; i++)
    outMatrix[i] = x[i];
}


/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    assert(nrhs == 4);
    double *inMatrix, *label;       
    mwSize ncols, nrow;
    tot = 0, num = 0, output_size = 0;
    
    inMatrix = mxGetPr(prhs[0]);

    ncols = mxGetN(prhs[0]);
    nrow = mxGetM(prhs[0]);    
    label = mxGetPr(prhs[1]);

    sp_graph(plhs, inMatrix, ncols, nrow, label);
}