/*
- Functions for calculating correlation between every pair of input vectors
- Solution is using instruction-level parallelism, multithreading, and vector instructions to optimize for speed. Memory access pattern is
  also optimized.
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

    //intermediate results
    std::vector<double> result_(ny*ny, 0);
    constexpr double4_t d4_0 {
    0.0, 0.0, 0.0, 0.0

    };

    // elements per vector
    constexpr int nb = 4;

    // vectors per input row
    int na = (nx + nb - 1) / nb;

    // block size
    constexpr int nd = 8;

    // how many blocks of rows
    int nc = (ny + nd - 1) / nd;
    // number of rows after padding
    int ncd = nc * nd;

    // input data, padded, converted to vectors
    double4_t* vd = double4_alloc(ncd*na);

    // slice size
    constexpr int s = 520;

    // how many slices per row
    int ns = (na + s - 1) / s;
    std::vector<double> sums(ny, 0.0);

    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; ++kb) {
                int i = ka * nb + kb;
                vd[na*j + ka][kb] = i < nx ? data[nx*j + i] : 0.0;
                sums[j] += vd[na*j + ka][kb];
            }
        }
    }

    for (int j = ny; j < ncd; ++j) {
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; ++kb) {
                vd[na*j + ka][kb] = 0.0;
            }
        }
    }

    // normalize
    #pragma omp parallel for
    for(int j=0; j < ny; j++){

        double sum = sums[j];
        double sum_squared = 0.0;
        double mean = 0.0;

        mean = sum/nx;
        double4_t mean4 = {mean, mean, mean, mean};

        for (int ka = 0; ka < na-1; ++ka){
            double4_t x4 = vd[na*j + ka];
            double4_t xm = x4-mean4;
            xm = xm * xm;
            sum_squared += sum_double4_t(xm);
            }
        // boundary
        for(int kb=0; kb < nb; ++kb){
            int i = (na-1) * nb + kb;
            sum_squared += i < nx ? pow(vd[na*j + (na-1)][kb] - mean, 2.0) : 0.0;
        }
        double denom = sqrt (sum_squared);
        for (int ka = 0; ka < na; ++ka){
            for(int kb = 0; kb < nb; ++kb){
                int i = ka * nb + kb;
                vd[na*j + ka][kb] = i < nx ? (vd[na*j + ka][kb]-mean ) / denom : 0.0;
            }
        }
    }

    // for slice
    for(int ks = 0; ks < ns; ++ks){
        //check outer bound
        int b = ks*s+s < na ? ks*s+s : na;
        #pragma omp parallel for schedule(dynamic,2)
        for(int ic= 0; ic < nc; ++ic){
            for(int jc=ic;jc < nc; ++jc){
                double4_t vv[nd][nd];
                for(int id = 0; id < nd; ++id){
                    for(int jd = 0; jd < nd; ++jd){
                        vv[id][jd] = d4_0;
                    }
                }
                for(int ka = ks*s; ka < b; ++ka){

                            double4_t y0 = vd[na*(jc * nd + 0) + ka];
                            double4_t y1 = vd[na*(jc * nd + 1) + ka];
                            double4_t y2 = vd[na*(jc * nd + 2) + ka];
                            double4_t y3 = vd[na*(jc * nd + 3) + ka];


                            double4_t y4 = vd[na*(jc * nd + 4) + ka];
                            double4_t y5 = vd[na*(jc * nd + 5) + ka];
                            double4_t y6 = vd[na*(jc * nd + 6) + ka];
                            double4_t y7 = vd[na*(jc * nd + 7) + ka];


                            double4_t x0 = vd[na*(ic * nd + 0) + ka];
                            double4_t x1 = vd[na*(ic * nd + 1) + ka];
                            double4_t x2 = vd[na*(ic * nd + 2) + ka];
                            double4_t x3 = vd[na*(ic * nd + 3) + ka];

                            double4_t x4 = vd[na*(ic * nd + 4) + ka];
                            double4_t x5 = vd[na*(ic * nd + 5) + ka];
                            double4_t x6 = vd[na*(ic * nd + 6) + ka];
                            double4_t x7 = vd[na*(ic * nd + 7) + ka];

                            vv[0][0] += x0 * y0;
                            vv[0][1] += x0 * y1;
                            vv[0][2] += x0 * y2;
                            vv[0][3] += x0 * y3;
                            vv[0][4] += x0 * y4;
                            vv[0][5] += x0 * y5;
                            vv[0][6] += x0 * y6;
                            vv[0][7] += x0 * y7;

                            vv[1][0] += x1 * y0;
                            vv[1][1] += x1 * y1;
                            vv[1][2] += x1 * y2;
                            vv[1][3] += x1 * y3;
                            vv[1][4] += x1 * y4;
                            vv[1][5] += x1 * y5;
                            vv[1][6] += x1 * y6;
                            vv[1][7] += x1 * y7;

                            vv[2][0] += x2 * y0;
                            vv[2][1] += x2 * y1;
                            vv[2][2] += x2 * y2;
                            vv[2][3] += x2 * y3;
                            vv[2][4] += x2 * y4;
                            vv[2][5] += x2 * y5;
                            vv[2][6] += x2 * y6;
                            vv[2][7] += x2 * y7;

                            vv[3][0] += x3 * y0;
                            vv[3][1] += x3 * y1;
                            vv[3][2] += x3 * y2;
                            vv[3][3] += x3 * y3;
                            vv[3][4] += x3 * y4;
                            vv[3][5] += x3 * y5;
                            vv[3][6] += x3 * y6;
                            vv[3][7] += x3 * y7;

                            vv[4][0] += x4 * y0;
                            vv[4][1] += x4 * y1;
                            vv[4][2] += x4 * y2;
                            vv[4][3] += x4 * y3;
                            vv[4][4] += x4 * y4;
                            vv[4][5] += x4 * y5;
                            vv[4][6] += x4 * y6;
                            vv[4][7] += x4 * y7;

                            vv[5][0] += x5 * y0;
                            vv[5][1] += x5 * y1;
                            vv[5][2] += x5 * y2;
                            vv[5][3] += x5 * y3;
                            vv[5][4] += x5 * y4;
                            vv[5][5] += x5 * y5;
                            vv[5][6] += x5 * y6;
                            vv[5][7] += x5 * y7;

                            vv[6][0] += x6 * y0;
                            vv[6][1] += x6 * y1;
                            vv[6][2] += x6 * y2;
                            vv[6][3] += x6 * y3;
                            vv[6][4] += x6 * y4;
                            vv[6][5] += x6 * y5;
                            vv[6][6] += x6 * y6;
                            vv[6][7] += x6 * y7;

                            vv[7][0] += x7 * y0;
                            vv[7][1] += x7 * y1;
                            vv[7][2] += x7 * y2;
                            vv[7][3] += x7 * y3;
                            vv[7][4] += x7 * y4;
                            vv[7][5] += x7 * y5;
                            vv[7][6] += x7 * y6;
                            vv[7][7] += x7 * y7;

                        }//end ka

                    for(int id = 0; id < nd; ++id){
                        for(int jd = 0; jd < nd; ++jd){
                            int i = ic * nd + id;
                            int j = jc * nd + jd;
                            if(i < ny && j < ny){
                                result_[j+ i*ny] += sum_double4_t(vv[id][jd]);
                            }//end if
                        }
                    }
                }//end jc
                }//end ic
        }//end ks
        // transfer double result_ to float result
        #pragma omp parallel for
        for(int j=0; j < ny; ++j){
            for(int i=j; i < ny; ++i){
                result[i+j*ny] = result_[i+j*ny];
            }
        }
    std::free(vd);
}