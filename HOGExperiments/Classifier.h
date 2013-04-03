#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "svm.h"
#include "svm_common.h"

enum svm_type {LIBSVM, LIGHTSVM, BINARYSVM};
	
class Classifier {
private:
	svm_model *model_libsvm;
	MODEL *model_lightsvm;
	int hidden_neurons, len_feature;
	double **weights1, *weights2, *bias1, bias2;
public:
	Classifier() {}
	
	~Classifier() {}

	void loadModel( const char *modelfile, svm_type type );
	//void loadModel( const char *w1, const char* w2, const char* b1, const char * b2 );
	float run( float *vector, int length, svm_type type );
	float run( double* vector );
};

#endif //CLASSIFIER_H
