#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define SUCCESS 1
#define FAILURE -1
#define INNER 11
#define LEAF 10
#define TOTALCOL 30

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

__global__ void
Split_kernel(int N, double** input, int attribute, int* left, int* right, int* index_array) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double attr = (double)attribute;
    int found = 0;
    if (index < N){
        if(index_array[index] == 1){
            double* input_line = input[index];
            for(int i = 5; i < TOTALCOL; i++){
                if(input_line[i] == attr){
                    left[index] = 1;
                    found = 1;
                    break;
                }
            }
            if(found == 0){
                right[index] = 1;
            }
        }
    }
}



__global__ void
Cal_resd_pref_kernel(int N, double** input, double gamma, int* index_array) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double attr = (double)attribute;
    int found = 0;
    if (index < N){
        if(index_array[index] == 1){
            double* input_line = input[index];
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



Node* BuildTree_Naive(double** samples, int sample_size, int max_attribute, int height, int* index_array,
                    int* pos_count, int* neg_count, double* gamma_top, double* gamma_bottom){

    const int threadsPerBlock = 512;
    const int blocks = (sample_size + threadsPerBlock - 1) / threadsPerBlock;

    Node *ret_node = new Node();

    //left and right node index mask total size of sample_size, 1 as in, 0 as not in
    int* left_array;
    int* right_array;
    cudaMalloc((void **)&left_array, sizeof(int) * sample_size);
    cudaMalloc((void **)&right_array, sizeof(int) * sample_size);
    cudaMemset(left_array, 0x00, sizeof(int)*sample_size);
    cudaMemset(right_array, 0x00, sizeof(int)*sample_size);


    // calculate most common value
    Pre_MCV<<<blocks, threadsPerBlock>>>(sample_size, samples, pos_count, neg_count, gamma_top, gamma_bottom, index_array);

    Reduce_MCV<<<blokcs, threadsPerBlock>>>(sample_size, pos_count, neg_count, gamma_top, gamma_bottom);

    // host calculate the count and popular result
    int pos_count_sum = 0;
    int neg_count_sum = 0;
    double gamma_top_sum = 0;
    double gamma_bottom_sum = 0;

    int rounded_length = nextPow2(sample_size);
    cudaMemcpy(&pos_count_sum, pos_count + rounded_length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_count_sum, neg_count + rounded_length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_top_sum, gamma_top + rounded_length - 1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_bottom_sum, gamma_bottom + rounded_length - 1, sizeof(double), cudaMemcpyDeviceToHost);

    int ret = 0;
    int count = pos_count_sum;
    if(pos_count_sum > neg_count_sum){
        ret = SUCCESS;
        count = pos_count_sum;
    }else{
        ret = FAILURE;
        count = neg_count_sum;
    }


    //leaf condition
    if(count == samples.size() || height == g_max_height){
        ret_node->attr = ""; //NULL for a string
        ret_node->status = LEAF;
        ret_node->value = ret;
        ret_node->height = height;
        ret_node->left = NULL;
        ret_node->right = NULL;
        double gamma = gamma_top_sum / gamma_bottom_sum;
        ret_node->gamma = gamma;
        Cal_resd_pref_kernel<<<blocks, threadsPerBlock>>>(sample_size, samples, gamma, index_array);
        return ret_node;
    }

    //inner node condition
    ret_node->status = INNER;
    ret_node->height = height;
    ret_node->value = -1;

    //int Best_attribute(double** samples, int sample_size, int max_attribute, int* index_array)
    int split_attr = Best_attribute(samples, sample_size, max_attribute, index_array); //@TODO WAIT FOR IMPLEMENT
    if(split_attr.size() == 0){ //string is zero length, for error handling
        fprintf(stderr, "Error: cannot find best attribute\n");
        return NULL;
    }
    ret_node->attr = split_attr;

    //use left and right array as the mask for next level node
    Split_kernel<<<blocks, threadsPerBlock>>>(sample_size, samples, split_attr, left_array, right_array, index_array);


    ret_node->left = BuildTree_Naive(samples, sample_size, max_attribute, height+1, left_array,
                                    pos_count, neg_count, gamma_top, gamma_bottom);
    ret_node->right = BuildTree_Naive(samples, sample_size, max_attribute, height+1, right_array,
                                    pos_count, neg_count, gamma_top, gamma_bottom);

    cudaFree(left_array);
    cudaFree(right_array);

    return ret_node;

}
