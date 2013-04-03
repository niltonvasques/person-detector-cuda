#ifndef LINEAR_CLASSIFY_H
#define LINEAR_CLASSIFY_H
#include <iostream>

/**
 * Read output model file after svm learning and fills in bias and weight
 * vector. Only linear SVM output model file are supported.
 */
class LinearClassify {
    public:

    // The default person detection parameters are hard coded, i.e. 
    // 8x8 cell, 2x2 no. of cells in each block, 9 orientation bin etc.
    // This default constructor simply loads the learned classify model. The model is hard coded. 
    LinearClassify() ;

    // Loads the linear SVM model from file 
    LinearClassify(std::string& filename, const int verbose = 0) ;

    LinearClassify(const LinearClassify& ) ;

    LinearClassify& operator=(const LinearClassify& ) ;

    int length() const 
    { return length_; }

    double operator()(const double* desc) const ;

    float operator()(const float* desc) const ;

    ~LinearClassify() {
        delete[] linearwt_;
		delete[] linearwtf_;
    }

    public:
        int length_;
        double* linearwt_;
        double linearbias_;
		float *linearwtf_;
};


#endif