#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <math.h>
// #include <utility>

using namespace std;

#define SUCCESS 1
#define FAILURE 0
#define INNER 11
#define LEAF 10

struct Node{
    int status; // 10 as leaf. 11 as inner
    int value;
    int height;
    string attr;
    struct Node* left;
    struct Node* right;
};

int g_max_height = 0;

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

double GetEntropyFromData(vector<map<string, int>* > samples) {
    int count = 0;
    // int total = 0;

    for(int i = 0; i < samples.size(); i++){
        map<string, int>* inner = samples[i];
        int inner_val = inner->find("result")->second;
        if(inner_val == SUCCESS){
            count++;
        }
    }

    return GetEntropyFromCount((double)count, (double)samples.size());
}

map<string, double> getMutualInfo(vector<map<string, int>* > data,
                                  set<string> attributes) {
    double entropy = GetEntropyFromData(data);

    map<string, double> result;
    double totalCount = (double) data.size();

    for (set<string>::iterator it=attributes.begin();it!=attributes.end();it++) {
        string attr = *it;
        double posCount = 0.0;
        double posSucCount = 0.0;
        double negSucCount = 0.0;
        for(int i = 0; i < data.size(); i++) {
            map<string, int>* inner = data[i];
            if (inner->find(attr) != inner->end()) {
                posCount++;
                int class_val = inner->find("result")->second;
                if(class_val == SUCCESS){
                    posSucCount++;
                }
            } else {
                int class_val = inner->find("result")->second;
                if(class_val == SUCCESS){
                    negSucCount++;
                }
           }
        }

        double negCount = totalCount - posCount;
        double pPos = posCount / totalCount;
        double pNeg = negCount / totalCount;
        double posEntropy = GetEntropyFromCount(posSucCount, posCount);
        double negEntropy = GetEntropyFromCount(negSucCount, negCount);
        double expectation = pPos*posEntropy + pNeg*negEntropy;
        double infoGain = entropy - expectation;
        result.insert(pair<string, double>(attr, infoGain));
    }

    return result;
}

string Best_attribute(vector<map<string, int>* > samples, set<string> attributes) {
    map<string, double> mutualInfo = getMutualInfo(samples, attributes);

    double maximum = -1.0;
    string maxAttr;

    for (map<string, double>::iterator it=mutualInfo.begin(); it!=mutualInfo.end();++it) {

        double curInfo = it->second;

        if (curInfo > maximum) {
            maxAttr = it->first;
            maximum = curInfo;
        }
    }
    return maxAttr;
}


int Most_common_value(vector<map<string, int>* >&samples, int* count){
    if(samples.size() == 0){
        return -1;
    }
    vector<map<string, int>* > :: iterator it;
    int succ_count = 0;
    int fail_count = 0;
    for(it = samples.begin(); it != samples.end(); it++){
        map<string, int>* inner = (*it);
        int inner_val = inner->find("result")->second;
        if(inner_val == SUCCESS){
            succ_count++;
        }else{
            fail_count++;
        }
    }
    if(succ_count >= fail_count){
        *count = succ_count;
        return SUCCESS;
    }else if (fail_count > succ_count){
        *count = fail_count;
        return FAILURE;
    }
}


Node* BuildTree_Naive(vector<map<string, int>* > &samples, set<string> &attributes, int height){
    Node *ret_node = new Node();
    int count = 0;
    int ret = Most_common_value(samples, &count);
    if(ret < 0){
        fprintf(stderr, "Error: cannot build tree node due to zero samples\n");
    }


    //leaf condition
    if(count == samples.size() || attributes.size() == 0 || height == g_max_height){
        ret_node->attr = ""; //NULL for a string
        ret_node->status = LEAF;
        ret_node->value = ret;
        ret_node->height = height;
        ret_node->left = NULL;
        ret_node->right = NULL;
        return ret_node;
    }

    //inner node condition

    ret_node->status = INNER;
    ret_node->height = height;
    ret_node->value = -1;

    string split_attr = Best_attribute(samples, attributes);
    if(split_attr.size() == 0){ //string is zero length, for error handling
        fprintf(stderr, "Error: cannot find best attribute\n");
        return NULL;
    }
    ret_node->attr = split_attr;

    //construct left and right samples
    vector<map<string, int>* > left_samples;
    vector<map<string, int>* > right_samples;

    for(int i = 0; i < samples.size(); i++){
        map<string, int>* inner = samples[i];
        if(inner->find(split_attr) == inner->end()){
            left_samples.push_back(inner);
        }else{
            right_samples.push_back(inner);
        }
    }

    cout <<" left size " << left_samples.size()<< " right size " << right_samples.size() <<endl;
    attributes.erase(split_attr);

    ret_node->left = BuildTree_Naive(left_samples, attributes, height+1);
    ret_node->right = BuildTree_Naive(right_samples, attributes, height+1);


    return ret_node;

}



int Predict_result (vector<map<string, int>* > &samples, Node* tree){
    if(tree == NULL){
        cout << "Error: the tree node is NULL"<< endl;
        return -1;
    }
    if(tree->status == LEAF){
        // string id = "id";
        string predict = "predict";
        int value = tree->value;
        for(int i = 0; i < samples.size(); i++){
            map<string, int>* line = samples[i];
            line->insert(pair<string, int>(predict, value));
        }
        return 1;
    }else{
        vector<map<string, int>* > left_samples;
        vector<map<string, int>* > right_samples;
        string split_attr = tree->attr;
        for(int i = 0; i < samples.size(); i++){
            map<string, int>* inner = samples[i];
            if(inner->find(split_attr) == inner->end()){
                left_samples.push_back(inner);
            }else{
                right_samples.push_back(inner);
            }
        }
        int ret = 0;
        ret += Predict_result(left_samples, tree->left);
        ret += Predict_result(right_samples, tree->right);
        if(ret != 2){
            return -1;
        }else{
            return 1;
        }
    }
}




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

    vector<map<string, int>* > samples;
    set<string> attributes;

    for (string line; getline(input, line); )
    {
        map<string, int>* sample_line = new map<string,int>();
        while((pos = line.find(delimiter_1)) != string::npos){
            string token = line.substr(0, pos);
            // cout<<token<<" ";
            if((temp = token.find(delimiter_2)) != string::npos){
                string key = token.substr(0, temp);
                token.erase(0, temp+delimiter_2.length());
                int value = atoi(token.c_str());
                sample_line->insert(pair<string, int>(key, value));
                attributes.insert(key);
            }else{
                int result_val = atoi(token.c_str());
                sample_line->insert(pair<string, int>(result, result_val));
            }
            line.erase(0, pos + delimiter_1.length());
        }

        if((temp = line.find(delimiter_2)) != string::npos){
            string key = line.substr(0, temp);
            line.erase(0, temp+delimiter_2.length());
            int value = atoi(line.c_str());
            sample_line->insert(pair<string, int>(key, value));
            attributes.insert(key);
        }else{
            int result_val = atoi(line.c_str());
            sample_line->insert(pair<string, int>(result, result_val));
        }
        //push the line map into the vector
        samples.push_back(sample_line);
    }
    input.close();

    // -----------------testing -----------------
    // string test = "wind";
    // if(attributes.find(test) != attributes.end()){
    //     cout<<"testing the file"<<endl;
    //     attributes.erase(test);
    //     if(attributes.find(test) != attributes.end()){
    //         cout<< "WTF???"<<endl;
    //     }
    // }
    // cout<< samples.size()<< endl;
    // cout<< attributes.size()<< endl;
    //
    // map<string, int>* test_line_1 = samples[1];
    // for (map<string,int>::iterator it=test_line_1->begin(); it!=test_line_1->end(); ++it){
    //     cout << it->first << " => " << it->second << " ";
    // }
    // cout << endl;
    //
    // for (set<string>::iterator it=attributes.begin(); it!=attributes.end(); ++it){
    //     cout << *it << " ";
    // }
    // cout << endl;

    string parse = Best_attribute(samples, attributes);

    // call BuildTree_Naive()

    Node * tree = BuildTree_Naive(samples, attributes, 0);

    cout<< "parsing result : "<< parse << endl;

    vector<map<string, int>* > pred_samples;
    set<string> pred_attributes;
    int id_val = 0;
    string id = "id";

    for (string line; getline(predict, line); )
    {
        map<string, int>* sample_line = new map<string, int>();
        while((pos = line.find(delimiter_1)) != string::npos){
            string token = line.substr(0, pos);
            // cout<<token<<" ";
            if((temp = token.find(delimiter_2)) != string::npos){
                string key = token.substr(0, temp);
                token.erase(0, temp+delimiter_2.length());
                int value = atoi(token.c_str());
                sample_line->insert(pair<string, int>(key, value));
                pred_attributes.insert(key);
            }else{
                int result_val = atoi(token.c_str());
                sample_line->insert(pair<string, int>(result, result_val));
            }
            line.erase(0, pos + delimiter_1.length());
        }

        if((temp = line.find(delimiter_2)) != string::npos){
            string key = line.substr(0, temp);
            line.erase(0, temp+delimiter_2.length());
            int value = atoi(line.c_str());
            sample_line->insert(pair<string, int>(key, value));
            pred_attributes.insert(key);
        }else{
            cout<<"Error: the last one does not have : delimiter_2  ["<< line <<"]" << endl;
        }
        sample_line->insert(pair<string, int>(id, id_val));
        id_val++;
        //push the line map into the vector
        pred_samples.push_back(sample_line);
    }
    predict.close();

    int ret = Predict_result(pred_samples, tree);

    int total = 0;
    int match = 0;

    for(int i = 0; i < pred_samples.size(); i++){
        map<string, int>* inner = pred_samples[i];
        if(inner->find(pred) == inner->end()){
            cout<<"Error: cannot find predict result" <<endl;
        }else{
            if(inner->find(result) == inner->end()){
                cout<<"Error: cannot find observed result" <<endl;
            }else{
                int pred_val = inner->find(pred)->second;
                int obsv_val = inner->find(result)->second;
                if(pred_val == obsv_val){
                    match++;
                }
                total++;
            }
        }

    }
    float rate = (float)match / (float)total;
    cout<< " total " << total << " match " <<match << endl;
    //
    //################need to free all samples

}



int foo(vector<map<string, int> >&samples){
    string pred = "pred";
    int value = 1;
    for(int i = 0; i < samples.size(); i++){
        map<string, int> &line = samples[i];
        // line.insert(pair<string, int>(pred, value));
        line[pred] = value;
    }
}

void bar2(vector<map<string, int>* >&samples, int iter, int maxIter){

    string pred = "pred";
    int value = 1;
    if(iter == maxIter || samples.size() == 0){
        for(int i = 0; i < samples.size(); i++){
            map<string, int>* line = samples[i];
            line->insert(pair<string, int>(pred, value));
        }
        return;
    }

    vector<map<string, int>* >left;
    vector<map<string, int>* >right;

    for(int i = 0; i < samples.size(); i++){
        map<string, int>* line = samples[i];
        if(line->find("map1_1") != line->end()){
            int val = line->find("map1_1")->second;
            if(val > 10){
                left.push_back(line);
            }else{
                right.push_back(line);
            }
        }
    }

    bar2(left, iter+1, maxIter);
    bar2(right, iter+1, maxIter);
    return;
}

void bar(vector<map<string, int> >&samples, int iter, int maxIter){

    string pred = "pred";
    int value = 1;
    if(iter == maxIter || samples.size() == 0){
        for(int i = 0; i < samples.size(); i++){
            map<string, int> &line = samples[i];
            line.insert(pair<string, int>(pred, value));
        }
        return;
    }

    vector<map<string, int>* >left;
    vector<map<string, int>* >right;

    for(int i = 0; i < samples.size(); i++){
        map<string, int> &line = samples[i];
        if(line.find("map1_1") != line.end()){
            int val = line.find("map1_1")->second;
            if(val > 10){
                left.push_back(&line);
            }else{
                right.push_back(&line);
            }
        }
    }

    bar2(left, iter+1, maxIter);
    bar2(right, iter+1, maxIter);
    return;
}


int test (void){
    map<string, int> map1;
    map<string, int> map2;
    map<string, int> map3;

    Node *p = new Node();
    p->attr = "testing\n";

    vector<map<string, int> > maps;

    map1.insert(pair<string, int>("map1_1", 1));
    map1.insert(pair<string, int>("map1_2", 2));
    map1.insert(pair<string, int>("map1_3", 3));

    map2.insert(pair<string, int>("map1_1", 11));
    map2.insert(pair<string, int>("map1_2", 12));
    map2.insert(pair<string, int>("map1_3", 13));

    map3.insert(pair<string, int>("map1_1", 21));
    map3.insert(pair<string, int>("map1_2", 22));
    map3.insert(pair<string, int>("map1_3", 23));

    maps.push_back(map1);
    maps.push_back(map2);
    maps.push_back(map3);

    fprintf(stderr, "the size of maps %d\n", maps.size());


    vector<map<string, int> > :: iterator it;
    for(int i = 0; i < maps.size(); i++ ){
        map<string, int>& map_temp = maps[i];
        if(i == 1){
            map_temp["map1_1"] = -1; // should not use insert
        }
    }
    // foo(maps);
    bar(maps, 1, 3);

    for(it = maps.begin(); it != maps.end(); it++){
        map<string, int> inner = (*it);
        map<string, int>::iterator nested;
        for( nested = inner.begin(); nested != inner.end(); ++nested)
        {
            cout << nested->first << " " << nested->second << endl; //ERROR
        }
    }

    cout << p->attr;
    delete(p);

}
