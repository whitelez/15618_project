#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>

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

vector<Node*> buildTree_multiple(double* samples2, int size, int attributes, int* index_array2) {
    // initial model
    vector<Node*> result;
    const int threadsPerBlock = 512;
    const int blocks = size/threadsPerBlock + 1;

    double* samples;
    cudaMalloc((void **)&samples, sizeof(double) * size * 30);
    cudaMemset(samples, 0x00, sizeof(double) * size * 30);
    cudaMemcpy(samples, samples2, sizeof(double) * size * 30, cudaMemcpyHostToDevice);

    int* index_array;
    cudaMalloc((void **)&index_array, sizeof(int) * size);
    cudaMemset(index_array, 0x00, sizeof(int) * size);
    cudaMemcpy(index_array, index_array2, sizeof(int) * size, cudaMemcpyHostToDevice);

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

    cudaMemcpy(&pos_count_last, pos_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_count_last, neg_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_top_last, gamma_top[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_bottom_last, gamma_bottom[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);

    MCV_reduce(pos_count, neg_count, gamma_top, gamma_bottom, rounded_length);
    cudaThreadSynchronize();

    int pos_count_result = 0;
    int neg_count_result = 0;
    double gamma_top_result = 0.0;
    double gamma_bottom_result = 0.0;

    cudaMemcpy(&pos_count_result, pos_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_count_result, neg_count[rounded_length-1], sizeof(int),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_top_result, gamma_top[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamma_bottom_result, gamma_bottom[rounded_length-1], sizeof(double),
              cudaMemcpyDeviceToHost);

    pos_count_result += pos_count_last;
    neg_count_result += neg_count_last;
    gamma_top_result += gamma_top_last;
    gamma_bottom_result += gamma_bottom_last;

    int count = reg_count_result;
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

    for(int i = 0; i < size; i++){
        double* line = &samples[i*LINE_SIZE];
        line[F_IDX] = f_init;
        // Original value was written in parsing
        //double val = line->find("result")->second;
        //line->insert(pair<string, double>(original, val));
        double pre_val = line[RESULT_IDX];
        double exp_val = 2.0*pre_val*f_init;
        line[RESULT_IDX] = 2.0*pre_val / (1.0+exp(exp_val));

        /*
        map<string, double>::iterator it = line->find("result");
        if (it != line->end()) {
            //@TODO check if need 0 or -1
            // currently not derived for 0
            double exp_val = 2.0*val*f_init;
            it->second = 2.0*val / (1.0+exp(exp_val));
        }
        */
    }

    for (int i = 1; i < max_iters; i++) {
        Node* tree = BuildTree_Naive(samples, size, attributes, 0, idex_array,
                                      pos_count, neg_count, gamma_top, gamma_bottom);
        tree->init = f_init;
        // calculate new residuls and gamma value
        //
        // for(int i = 0; i < samples.size(); i++){
        //     map<string, double>* line = samples[i];
        //     //@TODO double value!!!!!!!!!
        //     line->insert(pair<string, double>(f_result, f_init));
        //     int val = line->find("result")->second;
        //     line->insert(pari<string, int>(original, val));
        //     //@TODO check if update is valid
        //     //@TODO double value!!!!!!!!!
        //     std::map<char, int>::iterator it = line.find("result");
        //     if (it != m.end()) {
        //         //@TODO check if need 0 or -1
        //         // currently not derived for 0
        //         double exp_val = 2.0*(double)val*f_init;
        //         it->second = 2.0*(double)val / (1.0+exp(exp_val));
        //     }
        // }

        result.push_back(tree);
    }
    return result;
}


//Helper function for ABS
__device__ double ABS(double value){
    if(value < 0){
        return -value;
    }else{
        return value;
    }
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
    double attr = (double)attribute;
    int found = 0;
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
Cal_predict(int N, double* samples, double gamma, double f_init, double level, int* index_array){

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



Node* BuildTree_Naive(double* samples, int sample_size, int max_attribute, int height, int* index_array,
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
    cudaThreadSynchronize();
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
    if(pos_count_sum > neg_count_sum){
        ret = SUCCESS;
        count = pos_count_sum;
    }else{
        ret = FAILURE;
        count = neg_count_sum;
    }


    //leaf condition
    if(count == sample_size || height == g_max_height){
        ret_node->attr = -1; //NULL for a string
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
    double split_attr = Best_attribute(samples, sample_size, max_attribute, index_array); //@TODO WAIT FOR IMPLEMENT

    ret_node->attr = split_attr;

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

    ret_node->left = BuildTree_Naive(samples, sample_size, max_attribute, height+1, left_array,
                                    pos_count, neg_count, gamma_top, gamma_bottom);
    ret_node->right = BuildTree_Naive(samples, sample_size, max_attribute, height+1, right_array,
                                    pos_count, neg_count, gamma_top, gamma_bottom);

    cudaFree(left_array);
    cudaFree(right_array);

    return ret_node;

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
        // string id = "id";
        string predict = "predict";
        //@TODO double value!!!!!

        double gamma = tree->gamma;

        Cal_predict<<<blocks, threadsPerBlock>>>(sample_size, samples, gamma, f_init, level, index_array);
        cudaThreadSynchronize();

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

double* Predict_multiple(double* samples, int sample_size, vector<Node*> forest, int* index_array2){
    const int threadsPerBlock = 512;
    const int blocks = (sample_size + threadsPerBlock - 1) / threadsPerBlock;
    int rounded_length = nextPow2(sample_size);


    int* index_array;
    cudaMalloc((void **)&index_array, sizeof(int)*sample_length);
    cudaMemset(index_array, 0x00, sizeof(int)*sample_length);
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

    // calculate most common value
    //WARNING:: I Changed the name
    Pre_MCV_kernel<<<blocks, threadsPerBlock>>>(sample_size, samples, pos_count, neg_count, gamma_top, gamma_bottom, index_array);
    cudaThreadSynchronize();
    MCV_reduce(rounded_length, pos_count, neg_count, gamma_top, gamma_bottom);
    cudaThreadSynchronize();

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
    if(pos_count_sum > neg_count_sum){
        ret = SUCCESS;
        count = pos_count_sum;
    }else{
        ret = FAILURE;
        count = neg_count_sum;
    }

    double y_sum = (double)count*(double)ret+((double)sample_size-(double)count)*((double)-ret);
    double y_ave = y_sum / (double)sample_size;
    double f_init = 0.5 * log2((1+y_ave)/(1-y_ave));

    cout<< "PM loop" << endl;
    for(int i = 0; i < trees.size(); i++){
        Predict_Naive(samples, sample_size, tree->left, f_init, i, index_array);
    }


    double* device_output
    cudaMalloc((void **)&device_output, sizeof(double) * sample_size);
    Export_Result<<<blocks, threadsPerBlock>>>(sample_size, samples, device_output);


    double* output = (double*) malloc(sizeof(double)*sample_size);
    memset(output, 0, sizeof(double)*sample_size);
    cudaMemcpy(output, device_output, sizeof(double)*sample_size, cudaMemcpyDeviceToHost);

    cudaFree(pos_count);
    cudaFree(neg_count);
    cudaFree(gamma_top);
    cudaFree(gamma_bottom);
    cudaFree(device_output);

    cout<< "PM end" << endl;
    return output;
}
