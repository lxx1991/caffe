#include "mex.h"
#include "string.h"
#include "assert.h"

#define MAX_SP 700
#define MAX_EDGE 20000000

struct edge
{
  int prev;
  int ending;
};
struct edge e[MAX_SP * MAX_SP];
bool visit[MAX_SP * MAX_SP];
int head[MAX_SP];
int output[MAX_EDGE][3];

int tot, num_sp, output_size;

const int dx[4] = {0, 0, -1, 1};
const int dy[4] = {1, -1, 0, 0};

void add_edge(int x1, int x2)
{
  e[tot].prev = head[x1];
  e[tot].ending = x2;
  head[x1] = tot++;
}


void sp_graph(mxArray *plhs[], double *sp, int n, int m, double *instance)
{
  tot = 0, num_sp = 0, output_size = 0;
  
  int i, j, k;
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      if (sp[i*m+j] > num_sp)
        num_sp = sp[i*m+j];
  num_sp = num_sp + 1;

  assert(num_sp <= MAX_SP);
  memset(head, 255, sizeof(int) * num_sp);
  memset(visit, 0, sizeof(bool) * num_sp * num_sp);

  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
    {
      int x1 = sp[i*m+j];
      for (k=0; k<4; k++)
        if (i + dx[k] >= 0 && i + dx[k] < n && j + dy[k] >= 0 && j + dy[k] < m)
        {
          int x2 = sp[(i+dx[k])*m+j+dy[k]];
          if (x1 != x2 && !visit[x1 * num_sp + x2])
          {
            visit[x1 * num_sp + x2] = true;
            add_edge(x1, x2);
          }
        }
    }

  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
    {
      int x1 = sp[i*m+j];
      for (k=head[x1]; k!=-1; k = e[k].prev)
      {
        output[output_size][0] = j*n+i;
        output[output_size][1] = x1;
        output[output_size][2] = e[k].ending;
        output_size++;
      }
    }

  plhs[0] = mxCreateDoubleMatrix(output_size, 3, mxREAL);
  double *outMatrix = mxGetPr(plhs[0]);
  for (i=0; i<output_size; i++)
  {
    outMatrix[i] = output[i][0];
    outMatrix[i + output_size] = output[i][1];
    outMatrix[i + output_size * 2] = output[i][2];
  }
}


/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    double *sp, *instance;       
    mwSize ncols, nrow;
    
    sp = mxGetPr(prhs[0]);
    ncols = mxGetN(prhs[0]);
    nrow = mxGetM(prhs[0]);    
    instance = mxGetPr(prhs[1]);

    sp_graph(plhs, sp, ncols, nrow, instance);
}