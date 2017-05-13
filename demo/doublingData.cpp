#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>

using namespace std;

//@TODO parse max_iters
int main(int argc, char** argv){
    if(argc < 3){
        fprintf(stdout, "Usage: [training_file] [testing_file]\n");
        return -1;
    }

    // fprintf(stdout, "argc %d\n", argc);
    // return 0;
    //parse file
    ifstream input(argv[1]);
    ofstream output(argv[2]);


    vector<string> data;

    for (string line; getline(input, line); )
    {
        data.push_back(line);
    }

    for(int i = 0; i < 75; i++){
        for(int j = 0; j < data.size(); j++){
            output<<data[j] << endl;
        }
    }

    input.close();
    output.close();
}
