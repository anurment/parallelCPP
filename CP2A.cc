/*
- Functions for calculating correlation between every pair of input vectors
- solution is exploiting instruction-level parallelism
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

void correlate(int ny, int nx, const float *data, float *result) {
    double sum = 0.0;
    double mean = 0.0;
    double sum_squared = 0.0;

    std::vector<double> data_normalized(ny*nx, 0);

    for(int y=0; y < ny; y++){
        sum = 0.0;
        sum_squared = 0.0;
        mean = 0.0;

        for(int x=0; x < nx; x++){
            sum += data[x + y*nx];
        }
        mean = sum/nx;

        for(int x=0; x < nx; x++){
            sum_squared += pow(data[x + y*nx]- mean, 2.0);
        }
        double denom = sqrt (sum_squared);
        for(int x=0; x < nx; x++){
            data_normalized[x + y*nx] = ( data[x + y*nx]-mean ) / denom;
        }
    }

    // padd input data
    constexpr int nb = 4;
    int na = (nx + nb - 1) / nb;
    int nab = na*nb;

    std::vector<double> d(ny*nab, 0);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            d[nab*j + i] = data_normalized[nx*j + i];
            //t[nab*j + i] = data_normalized[n*i + j];
        }
    }
    double result_ = 0.0;
    for(int i=0; i < ny; i++){
        for(int j=i; j < ny; j++){
            double c[nb];
            for (int kb = 0; kb < nb; ++kb) {
                c[kb] = 0;
            }
            for(int ka=0; ka < na; ka++){
                for(int kb=0; kb < nb; kb++){
                    double a =  d[nab*i + ka*nb + kb];
                    double b = d[nab*j + ka*nb + kb];
                    c[kb] = a * b;
                }
                for(int kb = 0; kb < nb/2; kb++){
                    double s = c[2*kb] + c[2*kb+1];
                    result_ += s;
                }
           }
           result[j + i*ny] = result_;
           result_ = 0.0;
        }
    }
}