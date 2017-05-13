#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <set>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include "CycleTimer.h"

using namespace std;

#define SUCCESS 1
#define FAILURE -1
#define INNER 11
#define LEAF 10
#define LINE_SIZE 30
#define ORI_IDX 0
#define RESULT_IDX 1
#define F_IDX 2
#define PRED_IDX 3
#define ID_IDX 4
#define ATT_START_IDX 5

/* The following code is from 15-418 Assignment 2 scan.cu
 * @Author Jack Dong
 */

/* Helper function to round up to a power of 2.
 */
 struct Node{
     int status; // 10 as leaf. 11 as inner
     int value;
     int height;
     double gamma; // gamma for leaf
     double init; // f_init value
     double attr;
     struct Node* left;
     struct Node* right;
 };



int g_max_height = 3;
int max_iters = 3;


static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}



//Helper function for ABS
__device__ double ABS(double value){
    if(value < 0){
        return -value;
    }else{
        return value;
    }
}

__device__ int
Binary_search(double* line, double value)
{
  int lower = ATT_START_IDX;
  int upper = LINE_SIZE;
  while(lower < upper)
  {
    int mid = lower + (upper-lower)/2;
    if (line[mid] == value) return mid;
    if (line[mid] > value) upper = mid;
    else {
      lower = mid + 1;
    }
  }
  return -1;
}


//I changed the name!!!!!!!!!!!!!
__global__ void
Pre_MCV_kernel(int N, double* input, int* pos_count, int* neg_count, double* gamma_top, double* gamma_bottom, int* index_array){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N){
        double* input_line = &(input[index*30]);
        if(index_array[index] == 1){
            if(input_line[1] > 0){
                pos_count[index] = 1;
            }else{
                neg_count[index] = 1;
            }
            double ABS_val = ABS(input_line[1]);
            double bot_val = ABS_val*(2.0 - ABS_val);

            gamma_top[index] = input_line[1];
            gamma_bottom[index] = bot_val;
        }

    }
}

__global__ void
Split_kernel(int N, double* input, double attribute, int* left, int* right, int* index_array) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N){
        if(index_array[index] == 1){
            double* input_line = &(input[index*30]);

            int res_idx = Binary_search(input_line, attribute);
            if(res_idx < 0){
                left[index] = 1;
            }else{
                right[index] = 1;
            }
        }
    }
}



__global__ void
Cal_resd_pref_kernel(int N, double* input, double gamma, int* index_array) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N){
        if(index_array[index] == 1){
            double* input_line = &(input[index*30]);
            double prev_f = input_line[2];
            double curr_f = prev_f + gamma;
            input_line[2] = curr_f;

            double orig_val = input_line[0];
            double exp_val = 2.0*orig_val*curr_f;
            double resd_val = 2.0*orig_val / (1.0 + exp(exp_val));
            input_line[1] = resd_val;
        }
    }
}

__global__ void
Cal_predict(int N, double* input, double gamma, double f_init, double level, int* index_array){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N){
        if(index_array[index] == 1){
            double* input_line = &(input[index*30]);
            if(level == 0){
                double predict_val = f_init + gamma;
                input_line[PRED_IDX] = predict_val;
            }else{
                input_line[PRED_IDX] += gamma;
            }
        }
    }

}


__global__ void
Export_result_kernel(int N, double* input, double* output){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N){
        double* input_line = &(input[index*30]);
        output[index] = input_line[PRED_IDX];
    }

}

__global__ void
Set_value_kernel(int N, double* input, double f_init) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N){
        double* input_line = &(input[index*30]);
        input_line[F_IDX] = f_init;

        double orig_val = input_line[RESULT_IDX];
        double exp_val = 2.0*orig_val*f_init;
        double resd_val = 2.0*orig_val / (1.0 + exp(exp_val));
        input_line[RESULT_IDX] = resd_val;
    }
}


__global__ void
Upsweep(int* result, int length, int twod, int twod1) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * twod1;
    if (index < length) {
        result[index + twod1 - 1] += result[index + twod -1];
    }
}

__global__ void
Downsweep(int* result, int length, int twod, int twod1) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * twod1;
    if (index < length) {
        int t = result[index + twod - 1];
        result[index + twod - 1] = result[index + twod1 - 1];
        result[index + twod1 - 1] += t;
    }
}

void Exclusive_scan(int length, int* device_result)
{

    const int threadsPerBlock = 512;
    const int size = nextPow2(length);

    // upsweep phase
    for (int twod = 1; twod < size; twod*=2) {
        int twod1 = twod * 2;
        int blocks = size/twod1/threadsPerBlock + 1;
        Upsweep<<<blocks, threadsPerBlock>>>(device_result, size, twod, twod1);
        cudaThreadSynchronize();
    }

    cudaMemset(&device_result[size-1], 0, sizeof(int));

    // downsweep phase
    for (int twod = size/2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        int blocks = size/twod1/threadsPerBlock + 1;
        Downsweep<<<blocks, threadsPerBlock>>>(device_result, size, twod, twod1);
        cudaThreadSynchronize();
    }
}


double GetEntropyFromCount(double dcount, double dtotal) {
    double entropy = 0.0;
    if (dcount == 0 || dcount == dtotal) {
        return entropy;
    } else {
        double dremain = (dtotal-dcount);
        double entropy1 = dcount / dtotal * log2(dtotal / dcount);
        double entropy2 = dremain / dtotal * log2(dtotal / dremain);
        return entropy1+entropy2;
    }
}

__global__ void
Pre_data_entropy(int N, double* input, int* pos_count, int* index_array) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        if (index_array[index] == 1) {
            double* input_line = &input[index*LINE_SIZE];
            if (input_line[RESULT_IDX] > 0) {
                pos_count[index] = 1;
            }
        }
    }
}

double GetEntropyFromData(double* samples, int sample_size, int sample_total, int* index_array) {
    const int threadsPerBlock = 512;
    const int blocks = sample_size/threadsPerBlock + 1;

    int rounded_length = nextPow2(sample_size);
    int *pos_count;

    cudaMalloc((void**)&pos_count, sizeof(int)*rounded_length);
    cudaMemset(pos_count, 0x00, sizeof(int)*rounded_length);

    Pre_data_entropy<<<blocks, threadsPerBlock>>>(sample_size, samples, pos_count, index_array);
    cudaThreadSynchronize();

    int pos_count_last = 0;
    cudaMemcpy(&pos_count_last, &pos_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);

    Exclusive_scan(rounded_length, pos_count);

    int pos_count_result = 0;
    cudaMemcpy(&pos_count_result, &pos_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    pos_count_result += pos_count_last;

    cudaFree(pos_count);
    return GetEntropyFromCount((double)pos_count_result, (double)sample_total);
}

__global__ void
Pre_best_attr(int N, double* input, int* pos_count, int* pos_succ_count,
              int* neg_succ_count, int* index_array, double attr) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        if (index_array[index] == 1) {
            double* input_line = &input[index*LINE_SIZE];

            int res_idx = Binary_search(input_line, attr);
            if(res_idx < 0){
                if (input_line[RESULT_IDX] > 0) {
                    neg_succ_count[index] = 1;
                }
            } else {
                if (input_line[RESULT_IDX] > 0) {
                    pos_succ_count[index] = 1;
                }
                pos_count[index] = 1;
            }
        }
    }
}

__global__ void
Best_attr_upsweep(int* pos_count, int* pos_succ_count, int* neg_succ_count, int length, int twod, int twod1) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * twod1;
    if (index < length) {
        pos_count[index + twod1 - 1] += pos_count[index + twod -1];
        pos_succ_count[index + twod1 - 1] += pos_succ_count[index + twod -1];
        neg_succ_count[index + twod1 - 1] += neg_succ_count[index + twod -1];
    }
}

__global__ void
Best_attr_downsweep(int* pos_count, int* pos_succ_count, int* neg_succ_count, int length, int twod, int twod1) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * twod1;
    if (index < length) {
        int t_pos = pos_count[index + twod - 1];
        pos_count[index + twod - 1] = pos_count[index + twod1 - 1];
        pos_count[index + twod1 - 1] += t_pos;

        int t_pos_succ = pos_succ_count[index + twod - 1];
        pos_succ_count[index + twod - 1] = pos_succ_count[index + twod1 - 1];
        pos_succ_count[index + twod1 - 1] += t_pos_succ;

        int t_neg_succ = neg_succ_count[index + twod - 1];
        neg_succ_count[index + twod - 1] = neg_succ_count[index + twod1 - 1];
        neg_succ_count[index + twod1 - 1] += t_neg_succ;
    }
}

void Best_attr_reduce(int* pos_count, int* pos_succ_count, int* neg_succ_count, int length)
{

    const int threadsPerBlock = 512;
    const int size = nextPow2(length);

    // upsweep phase
    for (int twod = 1; twod < size; twod*=2) {
        int twod1 = twod * 2;
        int blocks = size/twod1/threadsPerBlock + 1;
        Best_attr_upsweep<<<blocks, threadsPerBlock>>>(pos_count, pos_succ_count, neg_succ_count, size, twod, twod1);
        cudaThreadSynchronize();
    }

    cudaMemset(&pos_count[size-1], 0, sizeof(int));
    cudaMemset(&pos_succ_count[size-1], 0, sizeof(int));
    cudaMemset(&neg_succ_count[size-1], 0, sizeof(int));

    // downsweep phase
    for (int twod = size/2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        int blocks = size/twod1/threadsPerBlock + 1;
        Best_attr_downsweep<<<blocks, threadsPerBlock>>>(pos_count, pos_succ_count, neg_succ_count, size, twod, twod1);
        cudaThreadSynchronize();
    }
}



double Best_attribute(double* data, int sample_size, int sample_total, set<double>& attributes, int* index_array) {
    double entropy = GetEntropyFromData(data, sample_size, sample_total, index_array);

    double maximum = -1.0;
    double best_attr = 0.0;
    int rounded_length = nextPow2(sample_size);

    const int threadsPerBlock = 512;
    const int blocks = sample_size/threadsPerBlock + 1;

    int* pos_count;
    int* pos_succ_count;
    int* neg_succ_count;

    cudaMalloc((void**)&pos_count, sizeof(int)*rounded_length);
    cudaMalloc((void**)&pos_succ_count, sizeof(int)*rounded_length);
    cudaMalloc((void**)&neg_succ_count, sizeof(int)*rounded_length);

    cudaMemset(pos_count, 0x00, sizeof(int)*rounded_length);
    cudaMemset(pos_succ_count, 0x00, sizeof(int)*rounded_length);
    cudaMemset(neg_succ_count, 0x00, sizeof(int)*rounded_length);


    for (set<double>::iterator it=attributes.begin();it!=attributes.end();it++) {
        double cur_attr = (*it);
        Pre_best_attr<<<blocks, threadsPerBlock>>>(sample_size, data, pos_count,
                                            pos_succ_count, neg_succ_count,
                                            index_array, cur_attr);
        cudaThreadSynchronize();

        int pos_count_last = 0;
        int pos_succ_count_last = 0;
        int neg_succ_count_last = 0;
        cudaMemcpy(&pos_count_last, &pos_count[rounded_length-1], sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&pos_succ_count_last, &pos_succ_count[rounded_length-1], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&neg_succ_count_last, &neg_succ_count[rounded_length-1], sizeof(int), cudaMemcpyDeviceToHost);

        Best_attr_reduce(pos_count, pos_succ_count, neg_succ_count, rounded_length);

        int pos_count_result = 0;
        int pos_succ_count_result = 0;
        int neg_succ_count_result = 0;

        cudaMemcpy(&pos_count_result, &pos_count[rounded_length-1], sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&pos_succ_count_result, &pos_succ_count[rounded_length-1], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&neg_succ_count_result, &neg_succ_count[rounded_length-1], sizeof(int), cudaMemcpyDeviceToHost);

        pos_count_result += pos_count_last;
        pos_succ_count_result += pos_succ_count_last;
        neg_succ_count_result += neg_succ_count_last;

        double neg_count_result = (double)sample_total - (double)pos_count_result;
        double p_pos = (double)pos_count_result / (double)sample_total;
        double p_neg = (double)neg_count_result / (double)sample_total;
        double pos_entropy = GetEntropyFromCount((double)pos_succ_count_result, (double)pos_count_result);
        double neg_entropy = GetEntropyFromCount((double)neg_succ_count_result, (double)neg_count_result);
        double expectation = p_pos*pos_entropy + p_neg*neg_entropy;
        double cur_info = entropy - expectation;
        if (cur_info > maximum) {
            best_attr = cur_attr;
            maximum = cur_info;
        }


        cudaMemset(pos_count, 0x00, sizeof(int)*rounded_length);
        cudaMemset(pos_succ_count, 0x00, sizeof(int)*rounded_length);
        cudaMemset(neg_succ_count, 0x00, sizeof(int)*rounded_length);
    }
    cudaFree(pos_count);
    cudaFree(pos_succ_count);
    cudaFree(neg_succ_count);
    return best_attr;

}


__global__ void
MCV_upsweep(int* pos_count, int* neg_count, double* gamma_top, double* gamma_bottom,
            int length, int twod, int twod1) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * twod1;
    if (index < length) {
        pos_count[index + twod1 - 1] += pos_count[index + twod -1];
        neg_count[index + twod1 - 1] += neg_count[index + twod -1];
        gamma_top[index + twod1 - 1] += gamma_top[index + twod -1];
        gamma_bottom[index + twod1 - 1] += gamma_bottom[index + twod -1];
    }
}

__global__ void
MCV_downsweep(int* pos_count, int* neg_count, double* gamma_top,double* gamma_bottom,
              int length, int twod, int twod1) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * twod1;
    if (index < length) {
        int t_pos = pos_count[index + twod - 1];
        pos_count[index + twod - 1] = pos_count[index + twod1 - 1];
        pos_count[index + twod1 - 1] += t_pos;

        int t_neg = neg_count[index + twod - 1];
        neg_count[index + twod - 1] = neg_count[index + twod1 - 1];
        neg_count[index + twod1 - 1] += t_neg;

        double t_top = gamma_top[index + twod - 1];
        gamma_top[index + twod - 1] = gamma_top[index + twod1 - 1];
        gamma_top[index + twod1 - 1] += t_top;

        double t_bottom = gamma_bottom[index + twod - 1];
        gamma_bottom[index + twod - 1] = gamma_bottom[index + twod1 - 1];
        gamma_bottom[index + twod1 - 1] += t_bottom;
    }
}

void MCV_reduce(int* pos_count, int* neg_count, double* gamma_top,
                double* gamma_bottom, int length)
{

    const int threadsPerBlock = 512;
    const int size = nextPow2(length);

    // upsweep phase
    for (int twod = 1; twod < size; twod*=2) {
        int twod1 = twod * 2;
        int blocks = size/twod1/threadsPerBlock + 1;
        MCV_upsweep<<<blocks, threadsPerBlock>>>(pos_count, neg_count, gamma_top,
                                                 gamma_bottom, size, twod, twod1);
        cudaThreadSynchronize();
    }

    cudaMemset(&pos_count[size-1], 0, sizeof(int));
    cudaMemset(&neg_count[size-1], 0, sizeof(int));
    cudaMemset(&gamma_top[size-1], 0.0, sizeof(double));
    cudaMemset(&gamma_bottom[size-1], 0.0, sizeof(double));



    // downsweep phase
    for (int twod = size/2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        int blocks = size/twod1/threadsPerBlock + 1;
        MCV_downsweep<<<blocks, threadsPerBlock>>>(pos_count, neg_count, gamma_top,
                                                   gamma_bottom, size, twod, twod1);
        cudaThreadSynchronize();

    }
}


Node* BuildTree_Naive(double* samples, int sample_size, set<double>& attributes, int height, int* index_array,
                    int* pos_count, int* neg_count, double* gamma_top, double* gamma_bottom){

    const int threadsPerBlock = 512;
    const int blocks = (sample_size + threadsPerBlock - 1) / threadsPerBlock;
    int rounded_length = nextPow2(sample_size);

    Node *ret_node = new Node();

    cudaMemset(pos_count, 0x00, sizeof(int)*rounded_length);
    cudaMemset(neg_count, 0x00, sizeof(int)*rounded_length);
    cudaMemset(gamma_top, 0x00, sizeof(double)*rounded_length);
    cudaMemset(gamma_bottom, 0x00, sizeof(double)*rounded_length);

    // calculate most common value
    //WARNING:: I Changed the name
    Pre_MCV_kernel<<<blocks, threadsPerBlock>>>(sample_size, samples, pos_count, neg_count, gamma_top, gamma_bottom, index_array);
    cudaThreadSynchronize();

    MCV_reduce(pos_count, neg_count, gamma_top, gamma_bottom, rounded_length);

    // host calculate the count and popular result
    int pos_count_sum = 0;
    int neg_count_sum = 0;
    double gamma_top_sum = 0;
    double gamma_bottom_sum = 0;

    cudaMemcpy(&pos_count_sum, pos_count + rounded_length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_count_sum, neg_count + rounded_length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_top_sum, gamma_top + rounded_length - 1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_bottom_sum, gamma_bottom + rounded_length - 1, sizeof(double), cudaMemcpyDeviceToHost);

    int ret = 0;
    int count = pos_count_sum;
    int total = pos_count_sum + neg_count_sum;
    if(pos_count_sum > neg_count_sum){
        ret = SUCCESS;
        count = pos_count_sum;
    }else{
        ret = FAILURE;
        count = neg_count_sum;
    }


    //leaf condition
    if(count == total || attributes.size() == 0 || height == g_max_height){
        ret_node->attr = -1; //NULL for a string
        ret_node->status = LEAF;
        ret_node->value = ret;
        ret_node->height = height;
        ret_node->left = NULL;
        ret_node->right = NULL;
        double gamma = gamma_top_sum / gamma_bottom_sum;
        ret_node->gamma = gamma;
        Cal_resd_pref_kernel<<<blocks, threadsPerBlock>>>(sample_size, samples, gamma, index_array);
        cudaThreadSynchronize();
        return ret_node;
    }

    //inner node condition
    ret_node->status = INNER;
    ret_node->height = height;
    ret_node->value = -1;

    //int Best_attribute(double** samples, int sample_size, int max_attribute, int* index_array)
    double split_attr = Best_attribute(samples, sample_size, total, attributes, index_array); //@TODO WAIT FOR IMPLEMENT

    ret_node->attr = split_attr;
    attributes.erase(split_attr);

    //left and right node index mask total size of sample_size, 1 as in, 0 as not in
    int* left_array;  //could change to char* or bit*
    int* right_array;
    cudaMalloc((void **)&left_array, sizeof(int) * sample_size);
    cudaMalloc((void **)&right_array, sizeof(int) * sample_size);
    cudaMemset(left_array, 0x00, sizeof(int)*sample_size);
    cudaMemset(right_array, 0x00, sizeof(int)*sample_size);


    //use left and right array as the mask for next level node
    Split_kernel<<<blocks, threadsPerBlock>>>(sample_size, samples, split_attr, left_array, right_array, index_array);
    cudaThreadSynchronize();



    ret_node->left = BuildTree_Naive(samples, sample_size, attributes, height+1, left_array,
                                    pos_count, neg_count, gamma_top, gamma_bottom);
    ret_node->right = BuildTree_Naive(samples, sample_size, attributes, height+1, right_array,
                                    pos_count, neg_count, gamma_top, gamma_bottom);

    cudaFree(left_array);
    cudaFree(right_array);

    return ret_node;

}


vector<Node*> BuildTree_multiple(double* samples, int size, set<double>& attributes, int* index_array) {
    // initial model
    vector<Node*> result;
    const int threadsPerBlock = 512;
    const int blocks = size/threadsPerBlock + 1;


    int rounded_length = nextPow2(size);
    int* pos_count;
    int* neg_count;
    double* gamma_top;
    double* gamma_bottom;

    cudaMalloc((void**)&pos_count, sizeof(int)*rounded_length);
    cudaMalloc((void**)&neg_count, sizeof(int)*rounded_length);
    cudaMalloc((void**)&gamma_top, sizeof(double)*rounded_length);
    cudaMalloc((void**)&gamma_bottom, sizeof(double)*rounded_length);

    cudaMemset(pos_count, 0x00, sizeof(int)*rounded_length);
    cudaMemset(neg_count, 0x00, sizeof(int)*rounded_length);
    cudaMemset(gamma_top, 0x00, sizeof(double)*rounded_length);
    cudaMemset(gamma_bottom, 0x00, sizeof(double)*rounded_length);

    Pre_MCV_kernel<<<blocks, threadsPerBlock>>>(size, samples, pos_count, neg_count, gamma_top, gamma_bottom, index_array);
    cudaThreadSynchronize();

    int pos_count_last = 0;
    int neg_count_last = 0;
    double gamma_top_last = 0.0;
    double gamma_bottom_last = 0.0;

    cudaMemcpy(&pos_count_last, &pos_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_count_last, &neg_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_top_last, &gamma_top[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_bottom_last, &gamma_bottom[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);

    MCV_reduce(pos_count, neg_count, gamma_top, gamma_bottom, rounded_length);
    cudaThreadSynchronize();



    int pos_count_result = 0;
    int neg_count_result = 0;
    double gamma_top_result = 0.0;
    double gamma_bottom_result = 0.0;

    cudaMemcpy(&pos_count_result, &pos_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_count_result, &neg_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_top_result, &gamma_top[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_bottom_result, &gamma_bottom[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);


    pos_count_result += pos_count_last;
    neg_count_result += neg_count_last;
    gamma_top_result += gamma_top_last;
    gamma_bottom_result += gamma_bottom_last;

    int count = neg_count_result;
    int ret = FAILURE;

    if (pos_count_result > neg_count_result) {
        count = pos_count_result;
        ret = SUCCESS;
    }
    double y_sum = (double)count * (double)ret  +   ((double)size-(double)count)*(-(double)ret) ;
    double y_ave = y_sum / (double)size;
    double f_init = 0.5 * log2((1+y_ave)/(1-y_ave));


    string f_result = "preF";
    string original = "original";

    Set_value_kernel<<<blocks, threadsPerBlock>>>(size, samples, f_init);




    for (int i = 1; i < max_iters; i++) {
        Node* tree = BuildTree_Naive(samples, size, attributes, 0, index_array,
                                      pos_count, neg_count, gamma_top, gamma_bottom);
        tree->init = f_init;

        result.push_back(tree);
    }

    // cout<<"BM out" <<endl;

    // cudaFree(samples);
    // cudaFree(index_array);
    cudaFree(pos_count);
    cudaFree(neg_count);
    cudaFree(gamma_top);
    cudaFree(gamma_bottom);

    return result;
}




int Predict_Naive(double* samples, int sample_size, Node* tree, double f_init, int level, int* index_array){

    if(tree == NULL){
        cout << "Error: the tree node is NULL"<< endl;
        return -1;
    }

    const int threadsPerBlock = 512;
    const int blocks = (sample_size + threadsPerBlock - 1) / threadsPerBlock;

    int ret = 0;

    if(tree->status == LEAF){

        double gamma = tree->gamma;
        Cal_predict<<<blocks, threadsPerBlock>>>(sample_size, samples, gamma, f_init, level, index_array);
        cudaThreadSynchronize();
        return 1;

    }else{

        int* left_array;  //could change to char* or bit*
        int* right_array;
        cudaMalloc((void **)&left_array, sizeof(int) * sample_size);
        cudaMalloc((void **)&right_array, sizeof(int) * sample_size);
        cudaMemset(left_array, 0x00, sizeof(int)*sample_size);
        cudaMemset(right_array, 0x00, sizeof(int)*sample_size);

        double split_attr = tree->attr;

        Split_kernel<<<blocks, threadsPerBlock>>>(sample_size, samples, split_attr, left_array, right_array, index_array);
        cudaThreadSynchronize();

        ret += Predict_Naive(samples, sample_size, tree->left, f_init, level, left_array);
        ret += Predict_Naive(samples, sample_size, tree->right, f_init, level, right_array);

        if(ret != 2){
            return -1;
        }else{
            return 1;
        }

    }

}

double* Predict_multiple(double* samples2, int sample_size, vector<Node*> forest, int* index_array2){
    const int threadsPerBlock = 512;
    const int blocks = (sample_size + threadsPerBlock - 1) / threadsPerBlock;
    int rounded_length = nextPow2(sample_size);

    double* samples;
    cudaMalloc((void **)&samples, sizeof(double) * sample_size * 30);
    cudaMemset(samples, 0x00, sizeof(double) * sample_size * 30);
    cudaMemcpy(samples, samples2, sizeof(double) * sample_size * 30, cudaMemcpyHostToDevice);

    int* index_array;
    cudaMalloc((void **)&index_array, sizeof(int)*sample_size);
    cudaMemset(index_array, 0x00, sizeof(int)*sample_size);
    cudaMemcpy(index_array, index_array2,  sizeof(int)*sample_size, cudaMemcpyHostToDevice);

    int* pos_count;
    int* neg_count;
    double* gamma_top;
    double* gamma_bottom;

    cudaMalloc((void **)&pos_count, sizeof(int) * rounded_length);
    cudaMalloc((void **)&neg_count, sizeof(int) * rounded_length);
    cudaMalloc((void **)&gamma_top, sizeof(double) * rounded_length);
    cudaMalloc((void **)&gamma_bottom, sizeof(double) * rounded_length);

    cudaMemset(pos_count, 0x00, sizeof(int)*rounded_length);
    cudaMemset(neg_count, 0x00, sizeof(int)*rounded_length);
    cudaMemset(gamma_top, 0x00, sizeof(double)*rounded_length);
    cudaMemset(gamma_bottom, 0x00, sizeof(double)*rounded_length);

    // calculate most common value
    Pre_MCV_kernel<<<blocks, threadsPerBlock>>>(sample_size, samples, pos_count, neg_count, gamma_top, gamma_bottom, index_array);
    cudaThreadSynchronize();

    int pos_count_last = 0;
    int neg_count_last = 0;
    double gamma_top_last = 0.0;
    double gamma_bottom_last = 0.0;

    cudaMemcpy(&pos_count_last, &pos_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_count_last, &neg_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_top_last, &gamma_top[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_bottom_last, &gamma_bottom[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);


    MCV_reduce(pos_count, neg_count, gamma_top, gamma_bottom, rounded_length);

    int pos_count_result = 0;
    int neg_count_result = 0;
    double gamma_top_result = 0.0;
    double gamma_bottom_result = 0.0;

    cudaMemcpy(&pos_count_result, &pos_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_count_result, &neg_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_top_result, &gamma_top[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_bottom_result, &gamma_bottom[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);

    pos_count_result += pos_count_last;
    neg_count_result += neg_count_last;
    gamma_top_result += gamma_top_last;
    gamma_bottom_result += gamma_bottom_last;


    int ret = 0;
    int count = pos_count_result;
    if(pos_count_result > neg_count_result){
        ret = SUCCESS;
        count = pos_count_result;
    }else{
        ret = FAILURE;
        count = neg_count_result;
    }

    double y_sum = (double)count * (double)ret  +   ((double)sample_size-(double)count)*(-(double)ret) ;
    double y_ave = y_sum / (double)sample_size;
    double f_init = 0.5 * log2((1+y_ave)/(1-y_ave));

    // cout<< "PM loop" << endl;
    for(int i = 0; i < forest.size(); i++){
        Predict_Naive(samples, sample_size, forest[i], f_init, i, index_array);
    }


    double* device_output;
    cudaMalloc((void **)&device_output, sizeof(double) * sample_size);
    Export_result_kernel<<<blocks, threadsPerBlock>>>(sample_size, samples, device_output);


    double* output = (double*) malloc(sizeof(double)*sample_size);
    memset(output, 0, sizeof(double)*sample_size);
    cudaMemcpy(output, device_output, sizeof(double)*sample_size, cudaMemcpyDeviceToHost);


    cudaFree(samples);
    cudaFree(index_array);
    cudaFree(pos_count);
    cudaFree(neg_count);
    cudaFree(gamma_top);
    cudaFree(gamma_bottom);
    cudaFree(device_output);

    // cout<< "PM end" << endl;
    return output;
}


//@TODO parse max_iters
int main(int argc, char** argv){
    if(argc < 5){
        fprintf(stdout, "Usage: train [training_file] [testing_file] [max_height] [max_iteration]\n");
        return -1;
    }

    //parse file
    ifstream input(argv[1]);
    ifstream predict(argv[2]);

    g_max_height = atoi(argv[3]);
    max_iters = atoi(argv[4]);

    string delimiter_1 = " ";
    string delimiter_2 = ":";
    string result = "result";
    string pred = "predict";
    size_t pos, temp;

    vector<double* > samples;
    vector<double* > samples_2;
    vector<double> observe;
    set<double> attributes;

    int id = 0;

    for (string line; getline(input, line); )
    {
        double* sample_line = (double*)malloc(sizeof(double)*30);
        memset(sample_line, 0, sizeof(double)*30);
        int curr = 5;
        while((pos = line.find(delimiter_1)) != string::npos){
            string token = line.substr(0, pos);
            // cout<<token<<" ";
            if((temp = token.find(delimiter_2)) != string::npos){
                string key = token.substr(0, temp);
                int key_value = atoi(key.c_str());
                double value = (double)key_value;
                sample_line[curr] = value;
                curr++;
                if(curr == 30){
                    cout<< "Error: the attribute is more than 25"<<endl;
                    return -1;
                }
                attributes.insert(value);
            }else{ //result
                int result_int_val = atoi(token.c_str());
                double result_val = (double)result_int_val;
                sample_line[0] = result_val; //original
                sample_line[1] = result_val; //result
            }
            line.erase(0, pos + delimiter_1.length());
        }

        if((temp = line.find(delimiter_2)) != string::npos){
            string key = line.substr(0, temp);
            int key_value = atoi(line.c_str());
            double value = (double)key_value;
            sample_line[curr] = value;
            curr++;
            if(curr == 30){
                cout<< "Error: the attribute is more than 25"<<endl;
                return -1;
            }
            attributes.insert(value);
        }else{
            int result_int_val = atoi(line.c_str());
            double result_val = (double)result_int_val;
            sample_line[0] = result_val;
            sample_line[1] = result_val;
        }
        //push the line map into the vector
        sample_line[4] = id;
        id++;
        samples.push_back(sample_line);
    }
    input.close();

    double** sample_matrix = &samples[0];
    int sample_length = samples.size();

    double* sample_array = (double*)malloc(sizeof(double)*sample_length*30);
    for(int i = 0; i < sample_length; i++){
        double* line = sample_matrix[i];
        memcpy(sample_array+(i*30), line, sizeof(double)*30);
        free(line);
    }


    int* index_array = (int*) malloc(sizeof(int)*sample_length);

    for(int i = 0; i < sample_length; i++){
        index_array[i] = 1;
    }



    double Prag_startTime = CycleTimer::currentSeconds();

    double* samples2;
    cudaMalloc((void **)&samples2, sizeof(double) * sample_length * 30);
    cudaMemcpy(samples2, sample_array, sizeof(double) * sample_length * 30, cudaMemcpyHostToDevice);

    int* index_array2;
    cudaMalloc((void **)&index_array2, sizeof(int) * sample_length);
    cudaMemcpy(index_array2, index_array, sizeof(int) * sample_length, cudaMemcpyHostToDevice);

    double Prag_endTime = CycleTimer::currentSeconds();

    double Build_startTime = CycleTimer::currentSeconds();
    vector<Node*> forest = BuildTree_multiple(samples2, sample_length, attributes, index_array2);
    double Build_endTime = CycleTimer::currentSeconds();

    cudaFree(samples2);
    cudaFree(index_array2);

    for (string line; getline(predict, line); )
    {
        double* sample_line = (double*)malloc(sizeof(double)*30);
        memset(sample_line, 0, sizeof(double)*30);
        int curr = 5;
        while((pos = line.find(delimiter_1)) != string::npos){
            string token = line.substr(0, pos);
            if((temp = token.find(delimiter_2)) != string::npos){
                string key = token.substr(0, temp);
                int key_value = atoi(key.c_str());
                double value = (double)key_value;
                sample_line[curr] = value;
                curr++;
                if(curr == 30){
                    cout<< "Error: the attribute is more than 25"<<endl;
                    return -1;
                }
            }else{ //result
                int result_int_val = atoi(token.c_str());
                double result_val = (double)result_int_val;
                sample_line[0] = result_val; //original
                sample_line[1] = result_val; //result
            }
            line.erase(0, pos + delimiter_1.length());
        }

        if((temp = line.find(delimiter_2)) != string::npos){
            string key = line.substr(0, temp);

            int key_value = atoi(line.c_str());
            double value = (double)key_value;
            sample_line[curr] = value;
            curr++;
            if(curr == 30){
                cout<< "Error: the attribute is more than 25"<<endl;
                return -1;
            }
        }else{
            int result_int_val = atoi(line.c_str());
            double result_val = (double)result_int_val;
            sample_line[0] = result_val;
            sample_line[1] = result_val;
        }
        //push the line map into the vector
        sample_line[4] = id;
        id++;
        samples_2.push_back(sample_line);
    }
    predict.close();

    double** sample_matrix_2 = &samples_2[0];
    int sample_length_2 = samples_2.size();

    double* sample_array_2 = (double*)malloc(sizeof(double)*sample_length_2*30);
    for(int i = 0; i < sample_length_2; i++){
        double* line = sample_matrix_2[i];
        memcpy(sample_array_2+(i*30), line, sizeof(double)*30);
        observe.push_back(line[ORI_IDX]);
        free(line);
    }

    int* index_array_2 = (int*) malloc(sizeof(int)*sample_length_2);

    for(int i = 0; i < sample_length_2; i++){
        index_array_2[i] = 1;
    }


    double Pred_startTime = CycleTimer::currentSeconds();
    double* predict_ret = Predict_multiple(sample_array_2, sample_length_2, forest, index_array_2);
    double Pred_endTime = CycleTimer::currentSeconds();


    int total = 0;
    int match = 0;

    for(int i = 0; i < observe.size(); i++){
        double pred_val = predict_ret[i];
        double obsv_val = observe[i];
        if(pred_val * obsv_val >= 0){
            match++;
        }
        total++;

    }
    float rate = (float)match / (float)total;

    cout<< "--------------------------------------------------------------"<<endl;
    cout<< "Traning Sample Size " << samples.size() <<endl;
    cout<< "Testing Sample Size " << samples_2.size() <<endl;
    cout<< "     Predict Match " << match << endl;
    cout<< "     Predict Rate " << rate << endl;
    cout<< "Total Time " << Prag_endTime - Prag_startTime + Build_endTime - Build_startTime + Pred_endTime - Pred_startTime <<endl;
    cout<< "     Data Copy Time " << Prag_endTime - Prag_startTime <<endl;
    cout<< "     Model Building Time " << Build_endTime - Build_startTime <<endl;
    cout<< "     Model Predict Time " << Pred_endTime - Pred_startTime <<endl;
    cout<< "--------------------------------------------------------------"<<endl;


    free(sample_array);
    free(sample_array_2);
    free(index_array);
    free(index_array_2);

    return 0;

}
