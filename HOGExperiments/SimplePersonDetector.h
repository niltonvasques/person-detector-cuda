/*
 * SimplePersonDetector.h
 *
 *  Created on: Mar 14, 2012
 *      Author: grimaldo
 */

#ifndef SIMPLEPERSONDETECTOR_H_
#define SIMPLEPERSONDETECTOR_H_

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include "Slide.h"

#include "Detector.h"
#include "HOGVasques.h"
#include "HOGLuciano.h"
#include "Classifier.h"
#include "HOGOpenCv.h"
#include "HOGCuda.h"
#include "HOGLuciano.h"
#include "Classifier.h"
#include "CudaHOG.h"
#include "FastHOG.h"
#include "LinearClassify.h"

class SimplePersonDetector : public Detector
{

public:
	SimplePersonDetector()
	{
		m_Hog.setSVMDetector(m_Hog.getDefaultPeopleDetector());
	}

	virtual bool Predict(const cv::Mat3b& im)
	{
		//std::cout << im.cols << " " << im.rows << std::endl;
		assert(im.cols == 64 && im.rows == 128);

		std::vector<cv::Point> found_locations;
		m_Hog.detect(im, found_locations);

		//cv::imwrite("teste.png", im);
		//std::cin.get();

		assert(found_locations.size() <= 1);
		return found_locations.size() > 0;
	}

	virtual void PredictMultiWindow( const cv::Mat3b &im, std::vector<cv::Rect> &founds  )
	{	
		
	}

	virtual int PredictMultiscale( const cv::Mat3b &im, std::vector<cv::Rect> &founds, SlidingOptions &opts  ){ 
		cv::Mat3b m;
		im.copyTo(m);
		int nwindows = Multiscale( m, founds, *this, opts );
	}

	virtual void options( SlidingOptions &opts ) { 
		mOpts.copyTo( opts );
	};

	virtual bool isDetectorType( string type ){
		return type.compare(getDetectorName()) == 0;
	}

	virtual std::string getDetectorName(){
		return std::string("SimplePersonDetector");
	}

private:
	cv::HOGDescriptor m_Hog;
	SlidingOptions mOpts;
};

/* Implementation of Histogram Oriented of Gradients for Person Detector

	By Nilton Vasques

*/
class VasquesPersonDetector : public Detector
{

public:
	VasquesPersonDetector()
	{
	}

	virtual bool Predict(const cv::Mat3b& im)
	{
		//std::cout << im.cols << " " << im.rows << std::endl;
		assert(im.cols == 64 && im.rows == 128);

		cv::Mat3b roi = cv::Mat3b::zeros( im.size() );
		im.copyTo( roi );

		std::vector<Positives> found_locations = m_Hog.detect( roi.data, im.cols, im.rows, 3 );

		roi.release();

		//cv::imwrite("teste.png", im);
		//std::cin.get();

		assert(found_locations.size() <= 1);
		return found_locations.size() > 0;
	}

	virtual void PredictMultiWindow( const cv::Mat3b &im, std::vector<cv::Rect> &founds  )
	{	}

	virtual int PredictMultiscale( const cv::Mat3b &im, std::vector<cv::Rect> &founds, SlidingOptions &opts  ){ 
		cv::Mat3b m;
		im.copyTo(m);
		return Multiscale( m, founds, *this, opts );
	}

	virtual bool isDetectorType( string type ){
		return type.compare(getDetectorName()) == 0;
	}

	virtual std::string getDetectorName(){
		return std::string("VasquesPersonDetector");
	}

private:
	HOGVasques m_Hog;
};

/* Implementation of Histogram Oriented of Gradients for Person Detector

	By Luciano Rebouças

*/

static Classifier *svm = NULL;
class LucianoPersonDetector : public Detector
{

public:
	LucianoPersonDetector()
	{
		m_Hog = new HOG( 54, 108, 54, 108, CPU );
		if( svm == NULL ){
			svm = new Classifier();
			cout << "Loading Model Start" <<endl;
			svm->loadModel( "svm_hog_54x108.poly3", LIGHTSVM );
			cout << "Loading Model End" <<endl;
		}
	}

	~LucianoPersonDetector(){
		delete m_Hog;
	}

	virtual bool Predict(const cv::Mat3b& im)
	{
		//std::cout << im.cols << " " << im.rows << std::endl;
		assert(im.cols == 54 && im.rows == 108);
		cv::Mat3b roi = cv::Mat3b::zeros( im.size() );
		im.copyTo( roi );

		m_Hog->computeGradients( roi.data, roi.channels() );
		m_Hog->computeWindowFeatures( 0, 0 );

		roi.release();

		//cv::imwrite("teste.png", im);
		//std::cin.get();
		return svm->run( m_Hog->hog, m_Hog->hogSize, LIGHTSVM ) > 0;
	}

	virtual void PredictMultiWindow( const cv::Mat3b &im, std::vector<cv::Rect> &founds  )
	{
		cv::Size window(54, 108);
		SlidingOptions opts;
		opts.scale_factor = 1;
		opts.window       = cv::Size(window.width, window.height);
		opts.step         = cv::Size(window.width / 8, window.height / 8);
		SlideAll(
			im,
			opts,
			std::back_inserter(founds),
			*this
		);
	}

	virtual int PredictMultiscale( const cv::Mat3b &im, std::vector<cv::Rect> &founds, SlidingOptions &opts  ){ 
		cv::Mat3b m;
		im.copyTo(m);
		return Multiscale( m, founds, *this, opts );
	}
	
	virtual bool isDetectorType( string type ){
		return type.compare(getDetectorName()) == 0;
	}

	virtual std::string getDetectorName(){
		return std::string("LucianoPersonDetector");
	}


private:
	HOG *m_Hog;
};

class OpenCVPersonDetector : public Detector
{

public:
	OpenCVPersonDetector()
	{
		m_Hog.setSVMDetector(HOGDescriptor_::getDefaultPeopleDetector());
	}

	virtual bool Predict(const cv::Mat3b& im)
	{
		//std::cout << im.cols << " " << im.rows << std::endl;
		assert(im.cols == 64 && im.rows == 128);

		Mat gray;

		cvtColor( im, gray, CV_BGR2GRAY );

		std::vector<cv::Point> found_locations;
		m_Hog.detect(gray, found_locations);

		//cv::imwrite("teste.png", im);
		//std::cin.get();

		assert(found_locations.size() <= 1);
		return found_locations.size() > 0;
	}

	virtual void PredictMultiWindow( const cv::Mat3b &im, std::vector<cv::Rect> &founds  )
	{	}

	virtual int PredictMultiscale( const cv::Mat3b &im, std::vector<cv::Rect> &founds, SlidingOptions &opts  ){ 
		cv::Mat3b m;
		im.copyTo(m);
		return Multiscale( m, founds, *this, opts );
	}

	virtual bool isDetectorType( string type ){
		return type.compare(getDetectorName()) == 0;
	}

	virtual std::string getDetectorName(){
		return std::string("OpenCVPersonDetector");
	}


private:
	HOGDescriptor_ m_Hog;
};

class CudaPersonDetector : public Detector
{

public:
	CudaPersonDetector()
	{
		//m_Hog.setSVMDetector(HOGDescriptor_::getDefaultPeopleDetector());
	}

	virtual bool Predict(const cv::Mat3b& im)
	{
		//std::cout << im.cols << " " << im.rows << std::endl;
		assert(im.cols == 64 && im.rows == 128);

		Mat gray;
		cvtColor( im, gray, CV_BGR2GRAY );
		//Cuda::Image image( im.data, Cuda::Size( im.cols, im.rows ), 3 );
		Cuda::Device::GpuImage<uchar> image = Cuda::Device::GpuImage<uchar>( gray.data, Cuda::Size( im.cols, im.rows ), 1 );
		std::vector<Cuda::Point> found_locations;

		Cuda::HOGCuda::getInstance()->detect( image, found_locations );

		unsigned char* data = (unsigned char*) malloc( sizeof(unsigned char) * im.cols * im.rows );
		Cuda::Device::GpuImage<float> grad = Cuda::HOGCuda::getInstance()->getGrad();
		cudaMemcpy( data, grad.ptr(0), grad.length(), cudaMemcpyDeviceToHost );

		Mat1b qangle = Mat1b::zeros( im.rows, im.cols );
		qangle.data = (unsigned char*)data;
		//Verificando se esta recebendo os dados da imagem corretamente
		//cv::Mat3b teste = cv::Mat3b::zeros( im.size() );
		//teste.data = image.data;
		assert( data != NULL );
		imshow( "Teste", qangle );
		waitKey();

		qangle.release();

		//cv::imwrite("teste.png", im);
		//std::cin.get();

		image.release();
		gray.release();
		assert(found_locations.size() <= 1);
		return found_locations.size() > 0;
	}

	virtual void PredictMultiWindow( const cv::Mat3b &im, std::vector<cv::Rect> &founds  )
	{	}

	virtual bool isDetectorType( string type ){
		return type.compare(getDetectorName()) == 0;
	}

	virtual std::string getDetectorName(){
		return std::string("CudaPersonDetector");
	}

};

class CudaPersonDetectorByPart : public Detector
{

public:
	CudaPersonDetectorByPart()
	{
		mHog = new HOG( 54, 108, 54, 108,  GPU );
		if( svm == NULL ){
			svm = new Classifier();
			cout << "Loading Model Start" <<endl;
			svm->loadModel( "svm_hog_54x108.poly3", LIGHTSVM );
			cout << "Loading Model End" <<endl;
		}
	}

	CudaPersonDetectorByPart( int imageWidth, int imageHeight)
	{
		mHog = new HOG( 54, 108, imageWidth, imageHeight,  GPU );
		if( svm == NULL ){
			svm = new Classifier();
			cout << "Loading Model Start" <<endl;
			svm->loadModel( "svm_hog_54x108.poly3", LIGHTSVM );
			cout << "Loading Model End" <<endl;
		}
	}

	virtual bool Predict(const cv::Mat3b& im)
	{
		//std::cout << im.cols << " " << im.rows << std::endl;
		assert(im.cols == 54 && im.rows == 108);
		
		cv::Mat3b roi;
		im.copyTo( roi );

		mHog->computeGradients( roi.data, roi.channels() );
		mHog->computeWindowFeatures( 0, 0 );
		float score = svm->run( mHog->hog, mHog->hogSize, LIGHTSVM );
		
		roi.release();

		return score > 0;
	}

	virtual int PredictMultiscale( const cv::Mat3b &im, std::vector<cv::Rect> &founds, SlidingOptions &opts  ){ 
		cv::Mat3b m;
		im.copyTo(m);
		return Multiscale( m, founds, *this, opts );
	}

	virtual void PredictMultiWindow( const cv::Mat3b &im, std::vector<cv::Rect> &founds  )
	{	
		mHog->computeGradients( im.data, im.channels() );
		mHog->detect( founds, svm );

		/*cv::Size window(54, 108);
		SlidingOptions opts;
		opts.scale_factor = 1;
		opts.window       = cv::Size(window.width, window.height);
		opts.step         = cv::Size(window.width / 8, window.height / 8);
		SlideAll(
			im,
			opts,
			std::back_inserter(founds),
			*this
		);*/
	}

	virtual bool isDetectorType( string type ){
		return type.compare(getDetectorName()) == 0;
	}

	virtual std::string getDetectorName(){
		return std::string("CudaPersonDetectorByPart");
	}

private:
	HOG* mHog;

};

class CudaPersonDetectorOneThreadBlock : public Detector
{

public:
	CudaPersonDetectorOneThreadBlock()
	{
		if( svm == NULL ){
			svm = new Classifier();
			cout << "Loading Model Start" <<endl;
			svm->loadModel( "svm_hog_54x108.poly3", LIGHTSVM );
			cout << "Loading Model End" <<endl;
		}

		mHog = new CudaHOG( 54, 108, 54, 108,  false, svm );
		hogVector = new float*[ 1 ];
		hogVector[0] = new float[ mHog->getHOGVectorSize() ];
	}

	~CudaPersonDetectorOneThreadBlock(){
		delete svm;
		delete mHog;
		delete[] hogVector;
	}

	virtual bool Predict(const cv::Mat3b& im)
	{
		//std::cout << im.cols << " " << im.rows << std::endl;
		assert(im.cols == 64 && im.rows == 128);

		cv::Mat3b resize;
		cv::resize( im, resize, cv::Size( 54, 108 ) );
		cv::Mat3b roi = cv::Mat3b::zeros( im.size() );
		resize.copyTo( roi );
		
		mHog->extractFeatures( roi.data, roi.channels() );
		mHog->getHOGVectorN( hogVector );

		float score = svm->run( hogVector[0], mHog->getHOGVectorSize(), LIGHTSVM );
		//float score = 0;
		return score > 0;
	}

	virtual void PredictMultiWindow( const cv::Mat3b &im, std::vector<cv::Rect> &founds  )
	{	
		delete mHog;
		mHog = new CudaHOG( im.cols, im.rows, 54, 108,  false, svm );
		vector<CudaPoint> founds_;
		mHog->extractFeatures( im.data, im.channels() );
		mHog->getFoundLocations( founds_ );
		for( int i = 0; i < founds_.size(); i++ ){
			CudaPoint pt = founds_.at(i);
			founds.push_back(cv::Rect( pt.x, pt.y, 54, 108 ));
		}
		cout << mHog->getWindowsCount() << endl;
	}

	virtual int PredictMultiscale( const cv::Mat3b &im, std::vector<cv::Rect> &founds, SlidingOptions &opts  ){ 
		cv::Mat3b m;
		im.copyTo(m);
		return Multiscale( m, founds, *this, opts );
	}

	virtual bool isDetectorType( string type ){
		return type.compare(getDetectorName()) == 0;
	}

	virtual std::string getDetectorName(){
		return std::string("CudaPersonDetectorOneThreadBlock");
	}

private:
	CudaHOG* mHog;
	float **hogVector;

};


class FastHOGPersonDetector : public Detector
{

public:
	FastHOGPersonDetector()
	{
		classify = new LinearClassify();
		mHog = new FastHOG_::FastHOG( 768, 576, classify->linearbias_, classify->linearwtf_, classify->length() );
	}

	~FastHOGPersonDetector(){
		delete classify;
		delete mHog;
	}

	virtual bool Predict(const cv::Mat3b& im)
	{
		assert(im.cols == 64 && im.rows == 128);

		Mat im_;
		cvtColor( im, im_, CV_BGR2BGRA );
		
		mHog->BeginProcess( im_.data );
		mHog->EndProcess();

		im_.release();

		return mHog->getPositivesCount() > 0;
	}

	virtual void PredictMultiWindow( const cv::Mat3b &im, std::vector<cv::Rect> &founds  )
	{	}

	virtual int PredictMultiscale( const cv::Mat3b &im, std::vector<cv::Rect> &founds, SlidingOptions &opts  )
	{
		Mat im_;
		cvtColor( im, im_, CV_BGR2BGRA );
		
		//mHog->BeginProcess( im_.data );
		mHog->BeginProcess( im_.data, -1, -1, -1, -1, -1.0, -1.0, opts.scale_factor );
		mHog->EndProcess();

		im_.release();

		for( int i = 0; i < mHog->getPositivesCount(); i++ ){
			int x = mHog->getPositives()[i].x;
			int y = mHog->getPositives()[i].y;
			int width = mHog->getPositives()[i].width;
			int height = mHog->getPositives()[i].height;
			founds.push_back(cv::Rect( x, y, width, height ));
		}
		mOpts.scales_count = mHog->scaleCount;
		mOpts.number_windows = mHog->nwindows;

		return mHog->nwindows;
	}

	virtual void options( SlidingOptions &opts ) { 
		mOpts.copyTo( opts );
	};

	virtual bool isDetectorType( string type ){
		return type.compare(getDetectorName()) == 0;
	}

	virtual std::string getDetectorName(){
		return std::string("FastHOGPersonDetector");
	}

private:
	FastHOG_::FastHOG* mHog;
	LinearClassify *classify;
	float **hogVector;
	SlidingOptions mOpts;

};



#endif /* SIMPLEPERSONDETECTOR_H_ */
