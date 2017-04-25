#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <stdio.h>
#include <vector>
#include <stdlib.h>

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

// string Best_attribute(vector<map<string, int> > samples)
//

int Most_common_value(vector<map<string, int> >&samples, int* count){
    if(samples.size() == 0){
        return -1;
    }
    vector<map<string, int> > :: iterator it;
    int succ_count = 0;
    int fail_count = 0;
    for(it = samples.begin(); it != samples.end(); it++){
        map<string, int> inner = (*it);
        int inner_val = inner.find("result")->second;
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


Node* BuildTree_Naive(vector<map<string, int> > &samples, vector<string> &attributes, int height){
    Node *ret_node = new Node();

    int count = 0;
    int ret = Most_common_value(samples, &count);
    if(ret < 0){
        fprintf(stderr, "Error: cannot build tree node due to zero samples\n");
    }

    //leaf condition
    if(count == samples.size() || attributes.size() == 0 || height == g_max_height){
        ret_node->attr = ""; //null for a string
        ret_node->status = LEAF;
        ret_node->value = ret;
        ret_node->height = height;
        ret_node->left = null;
        ret_node->right = null;
        return ret_node;
    }

    //inner node condition

    ret_node->status = INNER;
    ret_node->height = height;
    ret_node->value = -1;

    string split_attr = Best_attribute(samples);
    if(split_attr.size() == 0){ //string is zero length, for error handling
        fprintf(stderr, "Error: cannot find best attribute\n");
        return null;
    }
    ret_node->attr = split_attr;


    vector<map<string, int> > left_samples;
    vector<map<string, int> > right_samples;

    for(int i = 0; i < samples.size(); i++){
        map<string, int> &inner = samples[i];
        if(inner.find(split_attr) == inner.end()){
            left_samples.push_back(inner);
        }else{
            right_samples.push_back(inner);
        }
    }

    vector<string> sub_attribute;

    for(int i = 0; i < sub_attribute.size(); i++){
        if(split_attr.compare(attributes[i])){
            continue;
        }else{
            sub_attribute.push_back(attributes[i]);
        }
    }

    ret_node->left = BuildTree_Naive(left_samples, sub_attribute, height+1);
    ret_node->right = BuildTree_Naive(right_samples, sub_attribute, height+1);


    return ret_node;

}

int main(int argc, char** argv){
    if(argc < 3){
        fprintf(stdout, "Usage: train [training_file] [max_height]\n");
        return -1;
    }

    // fprintf(stdout, "argc %d\n", argc);
    // return 0;
    //parse file

    //call BuildTree_Naive()

    //
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

    map2.insert(pair<string, int>("map2_1", 11));
    map2.insert(pair<string, int>("map2_2", 12));
    map2.insert(pair<string, int>("map2_3", 13));

    map3.insert(pair<string, int>("map3_1", 21));
    map3.insert(pair<string, int>("map3_2", 22));
    map3.insert(pair<string, int>("map3_3", 23));

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
