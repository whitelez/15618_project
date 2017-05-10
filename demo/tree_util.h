struct Node{
    int status; // 10 as leaf. 11 as inner
    int value;
    int height;
    double gamma; // gamma for leaf
    double init; // f_init value
    string attr;
    struct Node* left;
    struct Node* right;
};

int g_max_height = 0;
int max_iters = 3;
