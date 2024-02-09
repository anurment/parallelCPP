/*
- Functions for calculating correlation between every pair of input vectors
- Simple sequential baseline solution
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

    //normalize vectors
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

    // calculate matrix product Y = X * X_T
    double result_=0.0;
    for(int i=0; i < ny; i++){
        for(int j=i; j < ny; j++){
            for(int k=0; k < nx; k++){
                result_ += data_normalized[k + j*nx] * data_normalized[k + i*nx];
           }
           result[j + i*ny] = result_;
           result_ = 0.0;
        }
    }
}