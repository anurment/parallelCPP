/*
- Functions for calculating correlation between every pair of input vectors
- Solution is using OpenMP and multithreading for parallelization
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
    std::vector<double> data_normalized(ny*nx,0.0);

    //normalize vectors
    #pragma omp parallel for
    for(int y=0; y < ny; y++){

        double sum = 0.0;
        double sum_squared = 0.0;
        double mean = 0.0;

        for(int x=0; x < nx; x++){
            sum += data[x + y*nx];
        }
        mean = sum/nx;
        for(int x=0; x < nx; x++){
            sum_squared += pow(data[x + y*nx]- mean, 2.0);
        }
        double denom = 1 / sqrt (sum_squared);
        for(int x=0; x < nx; x++){
            data_normalized[x + y*nx] = ( data[x + y*nx]-mean ) * denom;
        }
    }

    // calculate matrix product Y = X * X_T
    //double result_=0.0;
    for(int j=0; j < ny; j++){
        #pragma omp parallel for
        for(int i=j; i < ny; i++){
            double tmp = 0.0;
            for(int k=0; k < nx; k++){
                tmp += data_normalized[k + j*nx] * data_normalized[k + i*nx];
           }
           result[i + j*ny] = tmp;
        }
    }
}