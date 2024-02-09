/*
- Functions for calculating correlation between every pair of input vectors
- Solution is parallelized with the help of vector operations that are used 
  to perform multiple usefull arithmetic operations with one instruction.
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

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

//sum 4 elements
static inline double sum_double4_t(double4_t c) {
    double s = (c[0] + c[1])+ (c[2] + c[3]);
    return s;
}


void correlate(int ny, int nx, const float *data, float *result) {

    double sum = 0.0;
    double mean = 0.0;
    double sum_squared = 0.0;

    constexpr double4_t d4_0 {
    0.0, 0.0, 0.0, 0.0

    };

    // elements per vector
    constexpr int nb = 4;

    // vectors per input row
    int na = (nx + nb - 1) / nb;

    // input data, padded, converted to vectors
    double4_t* vd = double4_alloc(ny*na);

    for (int j = 0; j < ny; ++j) {
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; ++kb) {
                int i = ka * nb + kb;
                vd[na*j + ka][kb] = i < nx ? data[nx*j + i] : 0.0;
            }
        }
    }

    for(int j=0; j < ny; j++){

        sum = 0.0;
        sum_squared = 0.0;
        mean = 0.0;
        double4_t sum_4 = d4_0;

        //sum for mean
        for(int ka=0; ka < na; ka++){
            sum_4 += vd[na*j + ka];
        }
        sum = sum_double4_t(sum_4);
        mean = sum/nx;

        // (x-mean) **2
        for (int ka = 0; ka < na; ++ka){
            for(int kb = 0; kb < nb; ++kb){
                int i = ka * nb + kb;
                sum_squared += i < nx ? pow(vd[na*j + ka][kb] - mean, 2.0) : 0.0;
            }
        }

        double denom = 1 / sqrt (sum_squared);
        for (int ka = 0; ka < na; ++ka){
            for(int kb = 0; kb < nb; ++kb){
                int i = ka * nb + kb;
                vd[na*j + ka][kb] = i < nx ? (vd[na*j + ka][kb]-mean ) * denom : 0.0;
            }
        }
    }


    // matrix product
    for(int i=0; i < ny; i++){

        for(int j=i; j < ny; j++){

            double4_t c = d4_0;
            double4_t sum_4 = d4_0;

            for(int ka=0; ka < na; ka++){

                double4_t a = vd[na*i + ka];
                double4_t b = vd[na*j + ka];
                c = a * b;
                sum_4 += c;

           }
           result[j + i*ny] = sum_double4_t(sum_4);

        }
    }

    std::free(vd);
}