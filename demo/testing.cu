#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
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

int g_max_height = 3;



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

double GetEntropyFromData(double* samples, int sample_size, int* index_array) {
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
    return GetEntropyFromCount((double)pos_count_result, (double)sample_size);
}

__global__ void
Pre_best_attr(int N, double* input, int* pos_count, int* pos_succ_count,
              int* neg_succ_count, int* index_array, double attr) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int found = 0;
    if (index < N) {
        if (index_array[index] == 1) {
            double* input_line = &input[index*LINE_SIZE];
            for(int i = ATT_START_IDX; i < LINE_SIZE; i++){
                if(input_line[i] == attr){
                    found = 1;
                    break;
                }
            }
            if(found == 0){
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



double Best_attribute(double* data, int sample_size, int max_attr, int* index_array) {
    double entropy = GetEntropyFromData(data, sample_size, index_array);
    cout<< entropy <<endl;
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

    for (int i = 0; i < max_attr; i++) {
        double cur_attr = (double)(i+1);
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

        double neg_count_result = (double)sample_size - (double)pos_count_result;
        double p_pos = (double)pos_count_result / (double)sample_size;
        double p_neg = (double)neg_count_result / (double)sample_size;
        double pos_entropy = GetEntropyFromCount(pos_succ_count_result, (double)pos_count_result);
        double neg_entropy = GetEntropyFromCount(neg_succ_count_result, (double)neg_count_result);
        double expectation = p_pos*pos_entropy + p_neg*neg_entropy;
        double cur_info = entropy - expectation;
        if (cur_info > maximum) {
            best_attr = cur_attr;
            maximum = cur_info;
        }
        cout<<"Max " << maximum <<" info " << cur_info << " attr " << cur_attr <<endl;

        cudaMemset(pos_count, 0x00, sizeof(int)*rounded_length);
        cudaMemset(pos_succ_count, 0x00, sizeof(int)*rounded_length);
        cudaMemset(neg_succ_count, 0x00, sizeof(int)*rounded_length);
    }
    cudaFree(pos_count);
    cudaFree(pos_succ_count);
    cudaFree(neg_succ_count);
    return best_attr;

}


__device__ double ABS(double value){
    if(value < 0){
        return -value;
    }else{
        return value;
    }
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

        int t_bottom = gamma_bottom[index + twod - 1];
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

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int found = 0;
    if (index < N){
        if(index_array[index] == 1){
            double* input_line = &(input[index*30]);
            for(int i = 5; i < LINE_SIZE; i++){
                if(input_line[i] == attribute){
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
                input_line[3] = predict_val;
            }else{
                input_line[3] += gamma;
            }
        }
    }

}

__global__ void testing(double* input) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double * pointer = &(input[index*30]);
    pointer[1] = ABS(pointer[1]);
}

__global__ void testing2(double* input) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    input[index] = -1;
}


//@TODO parse max_iters
int main(int argc, char** argv){
    if(argc < 4){
        fprintf(stdout, "Usage: train [training_file] [max_height] [testing_file]\n");
        return -1;
    }

    // fprintf(stdout, "argc %d\n", argc);
    // return 0;
    //parse file
    ifstream input(argv[1]);
    ifstream predict(argv[3]);

    g_max_height = atoi(argv[2]);

    string delimiter_1 = " ";
    string delimiter_2 = ":";
    string result = "result";
    string pred = "predict";
    size_t pos, temp;

    vector<double* > samples;
    int maxAttribute = 0;

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
                if(key_value > maxAttribute){
                    maxAttribute = key_value;
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
            // line.erase(0, temp+delimiter_2.length());
            int key_value = atoi(line.c_str());
            double value = (double)key_value;
            sample_line[curr] = value;
            curr++;
            if(curr == 30){
                cout<< "Error: the attribute is more than 25"<<endl;
                return -1;
            }
            if(key_value > maxAttribute){
                maxAttribute = key_value;
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


    int* index_array2 = (int*) malloc(sizeof(int)*sample_length);

    for(int i = 0; i < sample_length; i++){
        index_array2[i] = 1;
    }

    //testing

    int rounded_length = nextPow2(sample_length);
    double* samples2;
    int* index_array;
    cudaMalloc((void **)&samples2, sizeof(double)*sample_length*30);
    cudaMalloc((void **)&index_array, sizeof(int)*sample_length);
    cudaMemset(samples2, 0x00, sizeof(double)*sample_length*30);
    cudaMemset(index_array, 0x00, sizeof(int)*sample_length);
    cudaMemcpy(samples2, sample_array,  sizeof(double)*sample_length*30, cudaMemcpyHostToDevice);
    cudaMemcpy(index_array, index_array2,  sizeof(int)*sample_length, cudaMemcpyHostToDevice);


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

    const int threadsPerBlock = 512;
    const int blocks = (sample_length + threadsPerBlock - 1) / threadsPerBlock;

    Pre_MCV_kernel<<<blocks, threadsPerBlock>>>(sample_length, samples2, pos_count, neg_count, gamma_top, gamma_bottom, index_array);
    cudaThreadSynchronize();

    int pos_count_sum = 0;
    int neg_count_sum = 0;
    double gamma_top_sum = 0;
    double gamma_bottom_sum = 0;

    cudaMemcpy(&pos_count_sum, pos_count + 0, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_count_sum, neg_count + 0, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_top_sum, gamma_top + 0, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_bottom_sum, gamma_bottom + 0, sizeof(double), cudaMemcpyDeviceToHost);

    fprintf(stderr, "cuda sum %d\n",pos_count_sum);
    fprintf(stderr, "cuda sum %d\n",neg_count_sum);
    fprintf(stderr, "cuda sum %f\n",gamma_top_sum);
    fprintf(stderr, "cuda sum %f\n",gamma_bottom_sum);



    MCV_reduce(pos_count, neg_count, gamma_top, gamma_bottom, rounded_length);
    cudaThreadSynchronize();

    // int pos_count_sum = 0;
    // int neg_count_sum = 0;
    // double gamma_top_sum = 0;
    // double gamma_bottom_sum = 0;

    cudaMemcpy(&pos_count_sum, pos_count + rounded_length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_count_sum, neg_count + rounded_length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_top_sum, gamma_top + rounded_length - 1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_bottom_sum, gamma_bottom + rounded_length - 1, sizeof(double), cudaMemcpyDeviceToHost);


    int pot = 0;
    int neg = 0;
    double ga_top = 0.0;
    double ga_bot = 0.0;

    for(int i = 0; i < sample_length; i++){
        double* line = &(sample_array[i*30]);
        double ABS_val = 0;
        if(line[1] > 0){
            pot++;
            ABS_val = line[1];
        }else{
            neg++;
            ABS_val = -line[1];
        }

        double bot_val = ABS_val*(2.0 - ABS_val);

        ga_top += line[1];
        ga_bot += bot_val;
    }

    if(pot != pos_count_sum){
        fprintf(stderr, "Error: pot %d not equal to cuda sum %d\n", pot, pos_count_sum);
    }
    if(neg != neg_count_sum){
        fprintf(stderr, "Error: neg %d not equal to cuda sum %d\n", neg, neg_count_sum);
    }
    if(ga_top != gamma_top_sum){
        fprintf(stderr, "Error: top %f not equal to cuda sum %f\n", ga_top, gamma_top_sum);
    }
    if(ga_bot != gamma_bottom_sum){
        fprintf(stderr, "Error: bot %f not equal to cuda sum %f\n", ga_bot, gamma_bottom_sum);
    }


    int* left_array;  //could change to char* or bit*
    int* right_array;
    cudaMalloc((void **)&left_array, sizeof(int) * sample_length);
    cudaMalloc((void **)&right_array, sizeof(int) * sample_length);
    cudaMemset(left_array, 0x00, sizeof(int)*sample_length);
    cudaMemset(right_array, 0x00, sizeof(int)*sample_length);

    double split_attr = (double)9.0;

    Split_kernel<<<blocks, threadsPerBlock>>>(sample_length, samples2, split_attr, left_array, right_array, index_array);
    cudaThreadSynchronize();

    int indicator_left = 0;
    int indicator_right = 0;
    cudaMemcpy(&indicator_left, left_array + 3, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&indicator_right, right_array + 3, sizeof(int), cudaMemcpyDeviceToHost);
    fprintf(stderr, "Left %d\n", indicator_left);
    fprintf(stderr, "Right %d\n", indicator_right);


    double gamma = gamma_top_sum / gamma_bottom_sum;
    double f_pre_gamma = 0;
    double r_pre_gamma = 0;
    double f_after_gamma = 0;
    double r_after_gamma = 0;
    cudaMemcpy(&r_pre_gamma, samples2 + 1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&f_pre_gamma, samples2 + 2, sizeof(double), cudaMemcpyDeviceToHost);
    Cal_resd_pref_kernel<<<blocks, threadsPerBlock>>>(sample_length, samples2, gamma, index_array);
    cudaThreadSynchronize();
    cudaMemcpy(&r_after_gamma, samples2 + 1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&f_after_gamma, samples2 + 2, sizeof(double), cudaMemcpyDeviceToHost);


    fprintf(stderr, "f_pre %f, f_after %f, r_pre %f, r_after %f, gamma %f,\n", f_pre_gamma, f_after_gamma, r_pre_gamma, r_after_gamma, gamma);


    double prd_pre = 0;
    double prd_after = 0;
    cudaMemcpy(&prd_pre, samples2 + 3, sizeof(double), cudaMemcpyDeviceToHost);
    Cal_predict<<<blocks, threadsPerBlock>>>(sample_length, samples2, gamma, 10, 0, index_array);
    cudaThreadSynchronize();
    cudaMemcpy(&prd_after, samples2 + 3, sizeof(double), cudaMemcpyDeviceToHost);
    fprintf(stderr, "f_pre %f, f_after %f,\n",prd_pre, prd_after);


    cudaMemcpy(&prd_pre, samples2 + 3, sizeof(double), cudaMemcpyDeviceToHost);
    Cal_predict<<<blocks, threadsPerBlock>>>(sample_length, samples2, gamma, 10, 1, index_array);
    cudaThreadSynchronize();
    cudaMemcpy(&prd_after, samples2 + 3, sizeof(double), cudaMemcpyDeviceToHost);
    fprintf(stderr, "f_pre %f, f_after %f,\n",prd_pre, prd_after);

    double best_attr = Best_attribute(samples2, sample_length, maxAttribute, index_array);
    fprintf(stderr, "BestAttr %f,\n",best_attr);


    // cudaMemset(pos_count, 0x00, sizeof(int)*rounded_length);
    // cudaMemset(neg_count, 0x00, sizeof(int)*rounded_length);
    // cudaMemset(gamma_top, 0x00, sizeof(double)*rounded_length);
    // cudaMemset(gamma_bottom, 0x00, sizeof(double)*rounded_length);
    //
    // Pre_MCV_kernel<<<blocks, threadsPerBlock>>>(sample_length, samples2, pos_count, neg_count, gamma_top, gamma_bottom, index_array);
    // cudaThreadSynchronize();
    //
    // cudaMalloc((void **)&teting, sizeof(int) * rounded_length);
    // cudaMemset(teting, 0x00, sizeof(int)*rounded_length);
    //
    // Best_attr_reduce(pos_count, neg_count, teting, rounded_length);
    // int new_pos = 0;
    // int new_neg = 0;
    // int new_tet = 0;
    // cudaMemcpy(&new_pos, pos_count + 0, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&new_neg, neg_count + 0, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&new_tet, teting + 0, sizeof(int), cudaMemcpyDeviceToHost);
    //
    // fprintf(stderr, "cuda sum %d\n",pos_count_sum);
    // fprintf(stderr, "cuda sum %d\n",neg_count_sum);
    // fprintf(stderr, "cuda sum %f\n",gamma_top_sum);
    // fprintf(stderr, "cuda sum %f\n",gamma_bottom_sum);

    fprintf(stderr, "Success!\n");

    cudaFree(samples2);
    cudaFree(left_array);
    cudaFree(right_array);
    cudaFree(pos_count);
    cudaFree(neg_count);
    cudaFree(gamma_top);
    cudaFree(gamma_bottom);



    return 0;


}


int test(void){
    int sample_size = 3000000;

    int threadsPerBlock = 512;
    int blocks = ((sample_size/30) + threadsPerBlock - 1) / threadsPerBlock;
    int blocks2 = ((sample_size) + threadsPerBlock - 1) / threadsPerBlock;

    double* test;
    cudaMalloc((void **)&test, sizeof(double) * sample_size);
    cudaMemset(test, 0x00, sizeof(double)*sample_size);

    testing2<<<blocks2, threadsPerBlock>>>(test);

    testing<<<blocks, threadsPerBlock>>>(test);

    double* host = (double*)malloc(sizeof(double)*sample_size);

    cudaMemcpy(host, test, sizeof(double)*sample_size, cudaMemcpyDeviceToHost);

    for(int i = 1 ; i < sample_size; i+= 30){
        if(host[i] != 1){
            fprintf(stderr, "WTF!!! %d %f \n",i,  host[i]);
        }
    }
    fprintf(stderr, "Success!\n");


    return 0;
}
