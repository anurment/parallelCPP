/*
- 2D median filtering with a rectangular windows
- parallized with OpenMP
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
using namespace std;

void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
    #pragma omp parallel for
    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx ; i++){
            int hxmin = i-hx > 0 ? i-hx : 0;
            int hxmax = i+hx+1 < nx ? i+hx+1 : nx;
            int hymin = j-hy > 0 ? j-hy : 0;
            int hymax = j+hy+1 < ny ? j+hy+1 : ny;
            int wn = (hxmax-hxmin) * (hymax-hymin);
            std::vector<float> w(wn, 0.0);
            int k = 0;
            for(int y = hymin; y < hymax; y++){
                for(int x = hxmin; x < hxmax; x++){
                    w[k] = in[x + y*nx];
                    k++;
                }
            }
            if( wn % 2 > 0){
                std::nth_element(w.begin(), w.begin() + wn/2, w.end());
                out[i + j*nx] = w[wn / 2];

            } else{
                std::nth_element(w.begin(), w.begin() + wn/2, w.end());
                float a = w[wn / 2];
                std::nth_element(w.begin(), w.begin() + wn/2 - 1, w.end());
                float b = w[wn / 2 -1];
                out[i + j*nx] = (a+b) / 2;
            }
        }
    }
}