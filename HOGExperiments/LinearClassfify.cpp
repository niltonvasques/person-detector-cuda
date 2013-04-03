#include <string>
#include <fstream>
#include <exception>
#include "LinearClassify.h"
#include "HOGVasques.h"
#include "persondetectorwt.tcc"

using namespace std;

extern const double         PERSON_WEIGHT_VEC[];
extern const int            PERSON_WEIGHT_VEC_LENGTH;
LinearClassify::LinearClassify() 
{// {{{
    linearbias_ = 6.6657914910925990525925044494215;
    length_ = PERSON_WEIGHT_VEC_LENGTH;
    linearwt_ = new double[length_];
    std::copy(PERSON_WEIGHT_VEC, PERSON_WEIGHT_VEC+PERSON_WEIGHT_VEC_LENGTH, linearwt_);
	linearwtf_ = new float[ length_ ];
	for( int i = 0 ; i < length_; linearwtf_[i] = linearwt_[i++] );
}// }}}

LinearClassify::LinearClassify(std::string& modelfile, const int verbose) 
    :
    length_ (0), linearwt_(0),  linearbias_(0)
{// {{{
    if (verbose > 2) 
        cout << "Reading model file: " << modelfile << endl;

    FILE *modelfl;
    if ((modelfl = fopen (modelfile.c_str(), "rb")) == NULL)
    { cout << "LinearClassify::LinearClassify " << " Unable to open the modelfile" << endl; }

    char version_buffer[10];
    if (!fread (&version_buffer,sizeof(char),10,modelfl))
    { cout <<"LinearClassify::LinearClassify" << " Unable to read version" << endl; }

    if(strcmp(version_buffer,"V6.01")) {
        cout <<"LinearClassify::LinearClassify" << " Version of model-file does not match version of svm_classify!" << endl; 
    }
    /* read version number */
    int version = 0;
    if (!fread (&version,sizeof(int),1,modelfl))
    { cout << "LinearClassify::LinearClassify"<< " Unable to read version number" << endl; }
    if (version < 200)
    { cout <<"LinearClassify::LinearClassify"<< " Does not support model file compiled for light version" << endl; }

    long kernel_type;
    fread(&(kernel_type),sizeof(long),1,modelfl);

    {// ignore these
        long poly_degree;
        fread(&(poly_degree),sizeof(long),1,modelfl);

        double rbf_gamma;
        fread(&(rbf_gamma),sizeof(double),1,modelfl);

        double  coef_lin;
        fread(&(coef_lin),sizeof(double),1,modelfl); 
        double coef_const;
        fread(&(coef_const),sizeof(double),1,modelfl);

        long l;
        fread(&l,sizeof(long),1,modelfl);
        char* custom = new char[l];
        fread(custom,sizeof(char),l,modelfl);
        delete[] custom;
    }

    long totwords;
    fread(&(totwords),sizeof(long),1,modelfl);

    {// ignore these
        long totdoc;
        fread(&(totdoc),sizeof(long),1,modelfl);

        long sv_num;
        fread(&(sv_num), sizeof(long),1,modelfl);
    }

    fread(&linearbias_, sizeof(double),1,modelfl);

    if(kernel_type == 0) { /* linear kernel */
        /* save linear wts also */
        linearwt_ = new double[totwords+1];
        length_ = totwords;
        fread(linearwt_, sizeof(double),totwords+1,modelfl);
    } else {
        cout << "LinearClassify::LinearClassify" << " Only supports linear SVM model files" << endl;
    }
    fclose(modelfl);

    if (verbose > 2) 
        std::cout << " Done" << std::endl;
}// }}}

LinearClassify::LinearClassify(const LinearClassify& o) :
    length_(o.length_), linearbias_(o.linearbias_)
{
    linearwt_ = new double[length_];
    std::copy(o.linearwt_, o.linearwt_+o.length_, linearwt_);
	linearwtf_ = new float[ length_ ];
	for( int i = 0 ; i < length_; linearwtf_[i] = linearwt_[i++] );
}

LinearClassify& LinearClassify::operator=(const LinearClassify& o) 
{
    if (&o != this) {
        if (linearwt_) 
            delete[] linearwt_;

        length_=o.length_; 
        linearbias_=o.linearbias_;

        linearwt_ = new double[length_];
        std::copy(o.linearwt_, o.linearwt_+o.length_, linearwt_);
    } 
    return *this;
}

double LinearClassify::operator()(const double* desc) const 
{
    double sum = 0;
    for (int i= 0; i< length_; ++i) 
        sum += linearwt_[i]*desc[i]; 
    return sum - linearbias_;
}

#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>

float LinearClassify::operator()(const float* desc) const 
{
    double sum = 0;
    for (int i= 0; i< length_; ++i) 
        sum += linearwt_[i]*desc[i]; 
	//if( sum - linearbias_ > 0 ){
	//	FILE *file = fopen("E://hog_features_dalal.txt","w+");
	//	if( file != NULL){
	//		fprintf(file,"Score: %f\n",sum-linearbias_);
	//		for( int i = 0; i< length_; i++){
	//			fprintf(file,"HOG[%d] - %f\n",i,desc[i]);
	//		}
	//		fclose(file);
	//	}
	//}
	//IplImage* image = cvCreateImage( cvSize( 64, 128 ), IPL_DEPTH_8U, 1 );
	//memset( image->imageData, 0, 64*128 );
	//HOGVasques hog;
	//hog.drawOrientationsBins( (unsigned char*)image->imageData, 64, 128, desc );
	//cvShowImage("Bins",image);
	//cvWaitKey();
	//cvReleaseImage( &image );

    return sum - linearbias_;
}