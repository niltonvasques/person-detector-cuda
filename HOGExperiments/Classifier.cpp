
#include "Classifier.h"

void Classifier::loadModel( const char* modelfile, svm_type type ) {
    
    if( type == LIBSVM ) {
	model_libsvm = svm_load_model( modelfile );
	//qDebug("LibSVM model loaded: %s", modelfile );
    }
    else if( type == LIGHTSVM ){
	model_lightsvm = read_model( (char *) modelfile );
	//qDebug("LightSVM model loaded: %s", modelfile );
    }
    
}
  
float Classifier::run( double* vector ) {
    
  //  double  sum1 = 0.0, sum2 = 0.0, 
  //                output[ hidden_neurons ];
  //   
  //   for( int h = 0; h < hidden_neurons; h++ ) {
	 //for( int w = 0; w < len_feature; w++ ) {
	 //    sum1 = sum1 + ( weights1[h][w] * vector[w] );
	 //    //cout << ( "%f", vector[w] ) << endl;
	 //}
  //   	 
	 ////qDebug( "Sum1 + bias = %f sum1 = %f bias = %f", sum1 + bias1[h], sum1, bias1[h] );
	 //output[h] = (double) ( 1.0 / ( 1.0 + exp( -( sum1 + bias1[h] ) ) ) );
	 //sum1 = 0.0;
	 ////qDebug( "OUTPUT[%d] = %f", h, output[h] );
  //   }
  //   
  //   
  //   for( int h = 0; h < hidden_neurons; h++ ) {
	 //sum2 = sum2 + ( weights2[h] * output[h] );
  //   }
  //   
  //   //qDebug("SAIDA= %f", sum2 + bias2 );
  //   return( sum2 + bias2 );
	return 0.0;
     
}

float Classifier::run( float* vector, int length, svm_type type ) {
    
    int i;
    double result;
        
    if( type == LIBSVM ) {
	//svm_node x[length + 1];
 //       
	//for( i = 0; i < length; i++ ) {
	//    x[i].index = i + 1;
	//    x[i].value = (double) vector[i];
	//}
	//x[i].index = -1;
 //   
	//result = svm_predict( model_libsvm, x );
    } else {
		register SVM_WORD *x;
		register DOC *doc;
		register char *comment = (char*) new char;
		x = (SVM_WORD *) new SVM_WORD[ length + 1 ];

		*comment = 0; 
        
		for( i = 0; i < length; i++ ) {
			x[i].wnum = i + 1;
			x[i].weight = (double) vector[i];
			//qDebug("[%ld] = %f", x[i].wnum, x[i].weight );
		}
		x[i].wnum = 0;
		
		int lenght  = 0;
		double *detector = new double[ 2160 ];
		memset( detector, 0, sizeof(double) * 2160 );
		//printf("ALPHA 0 %.16lf\n",model_lightsvm->alpha[0]);
		//printf("B %.16lf\n",model_lightsvm->b);
		//for( int j = 0; j < 2160; j++ ){
		//	for( int i = 1; i < model_lightsvm->sv_num; i++){
		//		double alpha = model_lightsvm->alpha[i];
		//		SVECTOR *vec = model_lightsvm->supvec[i]->fvec;
		//		SVM_WORD *word = vec->words;
		//		float weight = word[j].weight;
		//		detector[j] += alpha * weight;
		//	}
		//}

		//FILE *fOut = fopen( "E://hog_detector.txt","w+");
		//if( fOut != NULL ){
		//	fprintf(fOut, "const double hog_vector[] = { \n");
		//	for( int i = 0; i < 2160; i++){
		//		fprintf(fOut, " %.16lf%s\n",detector[i],(i == 2160-1 ? "}":",") );
		//	}
		//	fclose(fOut);
		//}
		
		doc = create_example( -1,0,0,0.0, create_svector( x, comment,1.0 ) );
	
		if( model_lightsvm->kernel_parm.kernel_type == 0 ) {
			add_weight_vector_to_linear_model( model_lightsvm );
		}
		result = classify_example( model_lightsvm, doc );	
		
		free( comment );
		free( x );
		free_example(doc,1);
    }
    
    return( result );
    
}

