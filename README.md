# 15618_project

This is the code base of the 15418/15618 final project : CuGB Cuda based Gradient Boosting Decision Tree

Code: In Demo folder, Gradient_boosting.cpp contains our sequential implementation of gradient boosting tree. cuda_main.cu contains our cuda parallel implementation of gradient boosting tree. 

Usage: 

    g++ doublingData.cpp -o multi

will provides a helper tool to extend files with multiple same contents

    g++ gradient_boosting.cpp -o seq_GB

Will provides sequential version of gradient boosting executable

    nvcc cuda_main.cu -o cuda_GB

will provide a cuda parallel version for gradient boosting executable


Data: in Data folder, it contains our demo data. It is in compressed version that only records binary attribuets that has positive result. Observation result is in the first column of the whole file. According to the property of decision tree, to test the performance, plase use multi tool to create a traning file that contains multiple original traning file, the result will stays the same.


For more information about our project, please [Check this website ](https://whitelez.github.io/cugb)


