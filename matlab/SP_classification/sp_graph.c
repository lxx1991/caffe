#include "mex.h"
#include "string.h"
#include "assert.h"

#define MAX_NUM 700
#define MAX_CLASS 100
#define MAX_INSTANCE 100

struct edge
{
  int prev;
  int ending;
};
struct edge e[MAX_NUM * MAX_NUM];
bool visit[MAX_NUM * MAX_NUM];
int q[MAX_NUM], dis[MAX_NUM], head[MAX_NUM];
int output[MAX_NUM * MAX_NUM][2];
int cnt_class[MAX_NUM*MAX_CLASS], cnt_instance[MAX_NUM*MAX_INSTANCE];
int sp_class[MAX_NUM], sp_instance[MAX_NUM];

int tot, num, output_size, num_class, num_instance;

void try_add_edge(int x1, int x2)
{
  if (!visit[x1 * num + x2])
  {
      visit[x1 * num + x2] = true;
      visit[x2 * num + x1] = true;

      e[tot].prev = head[x1];
      e[tot].ending = x2;
      head[x1] = tot++;

      e[tot].prev = head[x2];
      e[tot].ending = x1;
      head[x2] = tot++;
  }
}


void sp_graph(mxArray *plhs[], double *x, int n, int m, int step, double *label, double *instance)
{
  tot = 0, num = 0, output_size = 0, num_class = 0, num_instance = 0;
  int i, j, k;
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
    {
      if (label[i*m+j] > num_class)
        num_class = label[i*m+j];
      if (instance[i*m+j] > num_instance)
        num_instance = instance[i*m+j];
      if (x[i*m+j] > num)
        num = x[i*m+j];
    }
  
  num = num + 1;
  num_class = num_class + 1;
  num_instance = num_instance + 1;

  assert(num <= MAX_NUM);
  assert(num_class <= MAX_CLASS);
  assert(num_instance <= MAX_INSTANCE); 

  memset(head, 255, sizeof(int) * num);
  memset(visit, 0, sizeof(bool) * num * num);
  memset(cnt_class, 0, sizeof(int) * num * num_class);
  memset(cnt_instance, 0, sizeof(int) * num * num_instance);
  memset(sp_class, 0, sizeof(int) * num);
  memset(sp_instance, 0, sizeof(int) * num);

  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
    {
      cnt_class[(int)(x[i*m+j] * num_class + label[i*m+j])]++;
      cnt_instance[(int)(x[i*m+j] * num_instance + instance[i*m+j])]++;
    }
  for (i=0; i<num; i++)
  {
    for (j=1; j<num_class; j++)
      if (cnt_class[i * num_class + j] > cnt_class[i * num_class + sp_class[i]])
        sp_class[i] = j;
    for (j=1; j<num_instance; j++)
      if (cnt_instance[i * num_instance + j] > cnt_instance[i * num_instance + sp_instance[i]])
        sp_instance[i] = j;
  }

  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
    {
      int x1 = i*m+j, x2;
      if (i>0)
      {
        if (j>0)
        {
          x2 = (i-1)*m+(j-1);
          if (x[x1] != x[x2])
            try_add_edge(x[x1], x[x2]);
        }

        x2 = (i-1)*m+j;
        if (x[x1] != x[x2])
          try_add_edge(x[x1], x[x2]);

        if (j+1<m)
        {
          x2 = (i-1)*m+(j+1);
          if (x[x1] != x[x2])
            try_add_edge(x[x1], x[x2]);
        }
      }
      if (j>0)
      {
        x2 = i*m+(j-1);
        if (x[x1] != x[x2])
          try_add_edge(x[x1], x[x2]);
      }
    }
  for (i=0; i<num; i++)
  {
    memset(dis, 255, sizeof(int) * num);
    int h=0, t=0;
    q[0] = i; dis[i] = 0;
    while (h<=t)
    {
      if (dis[q[h]] < step)
      {
        for (j = head[q[h]]; j!=-1; j = e[j].prev)
          if (dis[e[j].ending] == -1)
          {
            dis[e[j].ending] = dis[q[h]] + 1;
            q[++t] = e[j].ending;
          }
      }

      if (q[h] > i)
      {
        output[output_size][0] = i;
        output[output_size++][1] = q[h];
      }
      h++;
    }
  }

  plhs[0] = mxCreateDoubleMatrix(output_size, 2, mxREAL);
  double *outMatrix = mxGetPr(plhs[0]);
  for (i=0; i<output_size; i++)
  {
    outMatrix[i] = output[i][0];
    outMatrix[i + output_size] = output[i][1];
  }


  plhs[1] = mxCreateDoubleMatrix(num, 1, mxREAL);
  outMatrix = mxGetPr(plhs[1]);
  for (i=0; i<num; i++)
    outMatrix[i] = sp_class[i];



  plhs[2] = mxCreateDoubleMatrix(num, 1, mxREAL);
  outMatrix = mxGetPr(plhs[2]);
  for (i=0; i<num; i++)
    outMatrix[i] = sp_instance[i];
}


/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    assert(nrhs == 4);
    double *inMatrix, *label, *instance;       
    mwSize ncols, nrow;
    tot = 0, num = 0, output_size = 0;
    
    inMatrix = mxGetPr(prhs[0]);

    ncols = mxGetN(prhs[0]);
    nrow = mxGetM(prhs[0]);    
    int step = (int)(*(mxGetPr(prhs[1])));
    label = mxGetPr(prhs[2]);
    instance = mxGetPr(prhs[3]);

    sp_graph(plhs, inMatrix, ncols, nrow, step, label, instance);
}