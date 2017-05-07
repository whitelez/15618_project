#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

#define SUCCESS 1
#define FAILURE -1
#define INNER 11
#define LEAF 10

int g_max_height = 0;

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
    int* index_array = (int*) malloc(sizeof(int)*sample_length);f

    //testing the reading of the file
    for(int i = 0; i < sample_length; i++){
        cout<<sample_matrix[i][0] << " ";
        index_array[i] = i;
    }
    cout <<endl;


    vector<Node*> BuildTree_multiple(double** sample_matrix, int sample_length, int max_Attribute, int* index_array);

    return 0;


}
