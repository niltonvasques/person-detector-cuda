#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>

#include "CudaHOG.h"
#include "gpu_utils.h"
#include "cuPrintf.cu"
#include "persondetectorwt.tcc"
#include "Classifier.h"

using namespace std;

extern const double         PERSON_WEIGHT_VEC[];
extern const int            PERSON_WEIGHT_VEC_LENGTH;

#define fmin( a, b ) ( (a)<(b)? (a):(b) )
template<typename T> inline bool isnan(T value){
	return value != value;
}
#define M_PI_DIV_2 2.5066282746310005024

__constant__ double SVM_VECTOR[PERSON_WEIGHT_VEC_LENGTH];
__constant__ float DESCRIPTOR_GAUSS[ 16 * 16 ];

float hostGaussian( float x, float y, float mx, float my, float sigma ) {
    float dist_, dx = x - mx, dy = y - my;
    dist_ =  sqrt( ( dx * dx ) + ( dy * dy ) );
    return exp( -dist_ * dist_ / ( sigma * sigma ) ) / ( M_PI_DIV_2 * sigma ); 
}

#define STEP_WIDTH	8
#define STEP_HEIGHT 16

CudaHOG::CudaHOG(int imageWidth_, int imageHeight_, int frameW_ , int frameH_ , bool oneThreadWindow_, Classifier *classifier_,  bool fullCircle_  ){
		assert( cudaFree( 0 ) == cudaSuccess );
		this->imageWidth	= imageWidth_;		
		this->imageHeight	= imageHeight_;
		this->frameW		= frameW_;
		this->frameH		= frameH_;
		//Melhores Valores encontrados por Dalal...
		this->blockStride	= 8;
		this->descWidth		= 16;
		this->descHeight	= 16;
		this->blockWidth	= 2;
		this->blockHeight	= 2;
		this->numHistBins	= 9;
		this->fullCircle	= fullCircle_;
		this->oneThreadWindow = oneThreadWindow_;
		this->classifier = classifier_;

		this->blocks = (( imageHeight - frameH )/ STEP_HEIGHT ) + 1;
		this->threads = (( imageWidth - frameW )/ STEP_WIDTH ) + 1;

		if( this->fullCircle ) this->histBinSpace = (int)( 360 / this->numHistBins );
		else this->histBinSpace = (int)( 180 / this->numHistBins );
	
		this->cellWidth = (int)( (float) descWidth / (float) blockWidth );
	
		///allocate final feature vector
		this->numBlocksW = int( (float) ( this->frameW - this->descWidth + this->blockStride ) / (float) this->blockStride );
		this->numBlocksH = int( (float) ( this->frameH - this->descHeight + this->blockStride ) / (float) this->blockStride );
		this->blockSize = this->numHistBins * this->blockWidth * this->blockHeight;
		this->hogSize =  this->blockSize * this->numBlocksW * this->numBlocksH;

		/* Alocando o vetor de Features HOG */
		size_t hogSizeWidth = this->hogSize * sizeof( float );
		size_t hogSizeHeight = blocks*threads;
		//cudaMalloc( (void**)&this->device_hog, hogSizeAlloc );
		assert( cudaMallocPitch( (void**)&this->device_hog, &hogS.pitch_hog, hogSizeWidth, hogSizeHeight) == cudaSuccess );
		assert( cudaMemset2D( this->device_hog, hogS.pitch_hog, 0, hogSizeWidth, hogSizeHeight ) == cudaSuccess );

		/* Alocando Magnitudes e Gradientes */
		size_t magGradSize	= sizeof( float ) * this->imageWidth * this->imageHeight ;
		assert( cudaMalloc( (void**)&this->device_mag, magGradSize ) == cudaSuccess );
		assert( cudaMalloc( (void**)&this->device_grad, magGradSize ) == cudaSuccess );
		assert( cudaMemset( this->device_mag, 0, magGradSize ) == cudaSuccess );
		assert( cudaMemset( this->device_grad, 0, magGradSize ) == cudaSuccess );

		//size_t descriptorWidth = sizeof(float) * this->numHistBins * this->blockHeight * this->blockWidth;
		//size_t descriptorHeight= blocks*threads;
		//if( oneThreadWindow ){
		//	assert( cudaMallocPitch( (void**)&device_desc, &hogS.pitch_descriptor, descriptorWidth, descriptorHeight ) == cudaSuccess );
		//	assert( cudaMemset2D( this->device_desc, hogS.pitch_descriptor, 0, descriptorWidth, descriptorHeight ) == cudaSuccess );
		//}

		assert( cudaMalloc( (void**)&gray, this->imageHeight * imageWidth ) == cudaSuccess );

		svm.linearbias_ = 6.6657914910925990525925044494215;
		//assert( cudaMalloc( (void**)&svm.linearwt_, PERSON_WEIGHT_VEC_LENGTH * sizeof(double) ) == cudaSuccess );
		cudaMemcpyToSymbol( SVM_VECTOR, PERSON_WEIGHT_VEC, PERSON_WEIGHT_VEC_LENGTH* sizeof(double) );
		assert( cudaMalloc( (void**)&svm.scores, blocks*threads * sizeof(double) ) == cudaSuccess );
		assert( cudaMemset( svm.scores, 0, blocks*threads * sizeof(double) ) == cudaSuccess );

		int* angle;
		assert( cudaMalloc( (void**)&angle, this->imageWidth * this->imageHeight * sizeof(int) ) == cudaSuccess );
		assert( cudaMemset( angle, 0, this->imageWidth * this->imageHeight * sizeof(int) ) == cudaSuccess );

		hogS.angle			= angle;
		hogS.blockHeight	= blockHeight;
		hogS.blockSize		= blockSize;
		hogS.blockStride	= blockStride;
		hogS.blockWidth		= blockWidth;
		hogS.cellWidth		= cellWidth;
		hogS.descHeight		= descHeight;
		hogS.descWidth		= descWidth;
		hogS.frameH			= frameH;
		hogS.frameW			= frameW;
		hogS.histBinSpace	= histBinSpace;
		hogS.imageHeight	= imageHeight;
		hogS.imageWidth		= imageWidth;
		hogS.numBlocksH		= numBlocksH;
		hogS.numBlocksW		= numBlocksW;
		hogS.numHistBins	= numHistBins;
		hogS.mag			= device_mag;
		hogS.grad			= device_grad;
		hogS.descriptor		= device_desc;
		hogS.hog			= device_hog;
		hogS.hogSize		= hogSize;
		hogS.bWHSize			= blockWidth * blockHeight;

		/* Foi percebido que o cálculo da Gaussiana estava se repetindo para todos os blocos, otimização, criando um vetor estático com os valores 
		 predefinidos da gaussiana */
		float gauss[ 16 * 16];
		for( int i = 0; i < 16; i++)
			for(int j = 0; j < 16; j++){
				gauss[i*16 +j] = hostGaussian( float(i), float(j), 8, 8, 8 );
				//printf("Gaussian[%d][%d] = %f\n",i,j,gauss[i*16+j]);
			}

		assert( cudaMemcpyToSymbol( DESCRIPTOR_GAUSS, gauss, 16*16*sizeof(float) ) == cudaSuccess );
		//cudaPrintfInit();
}

CudaHOG::~CudaHOG(){
	assert( cudaFree( this->hogS.angle ) == cudaSuccess );
	assert( cudaFree( this->hogS.mag ) == cudaSuccess );
	assert( cudaFree( this->hogS.grad ) == cudaSuccess );
	//if( this->oneThreadWindow ) assert( cudaFree( this->hogS.descriptor ) == cudaSuccess );
	assert( cudaFree( this->hogS.hog ) == cudaSuccess );
	assert( cudaFree( this->gray ) == cudaSuccess );
	assert( cudaFree( svm.scores ) == cudaSuccess );
	//cudaPrintfEnd();
	assert( cudaDeviceReset() == cudaSuccess );
}

__device__ float deviceGaussian( float x, float y, float mx, float my, float sigma ) {
    float dist_, dx = x - mx, dy = y - my;
    dist_ =  sqrt( ( dx * dx ) + ( dy * dy ) );
	//dist_ =  dx + dy;
    return exp( -dist_ * dist_ / ( sigma * sigma ) ) / ( M_PI_DIV_2 * sigma ); 
}

__device__ void deviceCircularInterpBin( DeviceHOG hog,  float value, int curBin, float *outCoef, int *outInterpBin ) {
  
    int halfSize = int( hog.histBinSpace >> 1 );
    
    if( value > halfSize ) { // range: (halfSize, binsize]
		*outInterpBin = ( curBin + 1 ) % hog.numHistBins;
		*outCoef = 1.0 - ( ( value - halfSize ) / hog.histBinSpace );
    } else { // range: [0, halfsize]
		*outInterpBin = ( curBin - 1 ) % hog.numHistBins;
		if( *outInterpBin < 0 ) *outInterpBin += hog.numHistBins;
		*outCoef = ( ( value + halfSize ) / hog.histBinSpace );
    }
}

#define DESC_AT_ELEMENT(desc,i,j,k) ( desc[ (( (i) * ( (hog.blockWidth) * (hog.blockHeight) ) + (j) * (hog.blockWidth) + (k)) ) ] )
__device__ void deviceCalculateL2Hys( DeviceHOG hog , float* descriptor) {

    float norm = 0.0, eps = 1.0;
    //compute norm
    for( int i = 0; i < hog.numHistBins; i++ )
	for( int j = 0; j < hog.blockHeight; j++ )
	    for( int k = 0; k < hog.blockWidth; k++ )
			norm += DESC_AT_ELEMENT( descriptor,i,j,k ) * DESC_AT_ELEMENT( descriptor,i,j,k );
    //L2-norm
    norm = sqrt( norm + eps ); 
    
    if ( !norm ) norm = 1.0;
    
    // Normalize and threshold ...
    for( int i = 0; i < hog.numHistBins; i++ )
	for( int j = 0; j < hog.blockHeight; j++ )
	    for( int k = 0; k < hog.blockWidth; k++ ) {
			DESC_AT_ELEMENT( descriptor,i,j,k ) /= norm;
			if( DESC_AT_ELEMENT( descriptor,i,j,k ) > 0.2 ) DESC_AT_ELEMENT( descriptor,i,j,k ) = 0.2;
		}
    
    norm = 0.0;
    for( int i = 0; i < hog.numHistBins; i++ )
	for( int j = 0; j < hog.blockHeight; j++ )
	    for( int k = 0; k < hog.blockWidth; k++ )
		norm += DESC_AT_ELEMENT( descriptor,i,j,k ) * DESC_AT_ELEMENT( descriptor,i,j,k );
    
    norm = sqrt( norm + eps );
    if ( !norm ) norm = 1.0;
    
    // and normalize again
    for( int i = 0; i < hog.numHistBins; i++ )
	for( int j = 0; j < hog.blockHeight; j++ )
	    for( int k = 0; k < hog.blockWidth; k++ )
		    DESC_AT_ELEMENT( descriptor,i,j,k ) /= norm;
    
}

__device__ void deviceWriteToVector( DeviceHOG hog, float *output, float* descriptor ) {
	float feat = 0.0;
	
	for( int b = 0; b < hog.numHistBins; b++ ) {
	    for( int y = 0; y < hog.blockHeight; y++ ) {
			for( int x = 0; x < hog.blockWidth; x++ ) {
				feat = DESC_AT_ELEMENT( descriptor,b,y,x );
				if( isnan( feat ) ) {
					feat = 0;
				}
				*output++ = feat;
			}
	    }
	}
}

__device__ void deviceComputeDescriptor( int bx, int by, DeviceHOG hog, float* descriptor ){
	float curGrad = 0.0, curMag = 0.0, gWeight = 1.0, 
            cellWeight = 1.0, binWeight = 0.0,
            dist = 0.0;
    int angle = 0, iBin = 0,
         stepx = 0, stepy = 0,
         dx = 0, dy = 0;
    
    for( int y = 0; y < hog.descHeight; y++ ) {
		for( int x = 0; x < hog.descWidth; x++ ) {
			int offset	= ((by+y)*hog.imageWidth) + bx+x;
			curGrad		= hog.grad[ offset ]; 
			curMag		= hog.mag[ offset ]; 
			angle		= hog.angle[ offset ];

			gWeight = DESCRIPTOR_GAUSS[ y * hog.descWidth + x ];
				   		    
			// histogram bin weighting
			iBin = 0; binWeight = 0;
			int halfSize = int( hog.histBinSpace >> 1 );
			float value = curGrad - hog.histBinSpace * angle;
			if( value > halfSize ) { // range: (halfSize, binsize]
				iBin = ( angle + 1 ) % hog.numHistBins;
				binWeight = 1.0 - ( ( value - halfSize ) / hog.histBinSpace );
			} else { // range: [0, halfsize]
				iBin = ( angle - 1 ) % hog.numHistBins;
				if( iBin < 0 ) iBin += hog.numHistBins;
				binWeight = ( ( value + halfSize ) / hog.histBinSpace );
			}

			int offset1_ = (angle) * ( hog.bWHSize );
			int offset2_ = (iBin) * ( hog.bWHSize );

			float gW_curMag = gWeight * curMag;
			float bin_gW_curMag = gW_curMag * binWeight;
			float oneMinusbin_gW_curMag = gW_curMag * ( 1.0 - binWeight );

			for( int iy = 0; iy < hog.blockHeight; iy++ ) {
				for( int ix = 0; ix < hog.blockWidth; ix++ ) {
					dx = x - ( 8 + ( ix << 3 ) );
					dy = y - ( 8 + ( iy << 3 ) );
					dist = sqrt( (float) ( dx * dx ) + ( dy * dy ) );  
					//dist = abs(dx + dy);
			//		//cell weighting
					cellWeight = 1.0 - fmin( (float) ( (float) dist / (float) hog.cellWidth ), (float) 1.0 );

					int offset1 = (offset1_ + (iy) * (hog.blockWidth) + (ix)) ;
					int offset2 = (( offset2_ + (iy) * (hog.blockWidth) + (ix)) );
					descriptor[offset1] += bin_gW_curMag * cellWeight;
					descriptor[offset2] += oneMinusbin_gW_curMag * cellWeight;					
				}
			}
		}
    }
}

#define BLOCK_DESCRIPTOR_SIZE 2*2*9
__global__ void cudaSlidingWindow( DeviceHOG hog, LINEAR_CLASSIFY_SVM svm ){
	int py	= blockIdx.x * STEP_HEIGHT;
	int px	= threadIdx.x * STEP_WIDTH;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//cuPrintf("LinearBias %f\n",svm.linearbias_);
	//float hog_row[ 3780 ];
	size_t descriptorSize = sizeof( float ) * hog.numHistBins * hog.blockHeight * hog.blockWidth ;
	if( (py < hog.imageHeight - hog.frameH + 1 ) && ( px < hog.imageWidth - hog.frameW + 1) ){
		int i = 0;
		float descriptor_row[BLOCK_DESCRIPTOR_SIZE];
		float* hog_row = (float*)((char*)hog.hog + tid * hog.pitch_hog);
		for( int by = 0; by <=  hog.frameH - hog.descHeight; by += hog.blockStride ) {
			for( int bx = 0; bx <=  hog.frameW - hog.descWidth; bx += hog.blockStride ) {
				memset( descriptor_row, 0, descriptorSize );
				deviceComputeDescriptor( bx+px, by+py, hog, descriptor_row );
				deviceCalculateL2Hys( hog ,descriptor_row);
				deviceWriteToVector( hog, &hog_row[i], descriptor_row );
				i += hog.blockSize;
			}
		}
		double sum = 0;
		for (int i= 0; i< hog.hogSize; ++i) 
				sum += SVM_VECTOR[i]*hog_row[i]; 
		 svm.scores[tid] = sum - svm.linearbias_;
		 //cuPrintf( "Score[%d] -  %f\n",tid,sum); 
	}
	
}

/* Uma thread por bloco hog */

__global__ void cudaHogThreadBlock( DeviceHOG hog, LINEAR_CLASSIFY_SVM svm ){
	int py	= blockIdx.y * STEP_HEIGHT;
	int px	= blockIdx.x * STEP_WIDTH;
	int tid = blockIdx.y * gridDim.x + blockIdx.x;

	int by = threadIdx.y * hog.blockStride;
	int bx = threadIdx.x * hog.blockStride;

	int i = (threadIdx.y * blockDim.x + threadIdx.x) * hog.blockSize;
	//cuPrintf("LinearBias %f\n",svm.linearbias_);
	//float hog_row[ 3780 ];
	size_t descriptorSize = sizeof( float ) * BLOCK_DESCRIPTOR_SIZE ;
	if( (py < hog.imageHeight - hog.frameH + 1 ) && ( px < hog.imageWidth - hog.frameW + 1) ){
		float descriptor_row[BLOCK_DESCRIPTOR_SIZE];
		float* hog_row = (float*)((char*)hog.hog + tid * hog.pitch_hog);
		memset( descriptor_row, 0, descriptorSize );
		deviceComputeDescriptor( bx+px, by+py, hog, descriptor_row );
		deviceCalculateL2Hys( hog ,descriptor_row);
		deviceWriteToVector( hog, &hog_row[i], descriptor_row );
		//	}
		//}
		//__syncthreads();
		//if( threadIdx.x == 0 && threadIdx.y == 0 ){
		//	double sum = 0;
		//	for (int i= 0; i< hog.hogSize; ++i) 
		//			sum += SVM_VECTOR[i]*hog_row[i]; 
		//	 svm.scores[tid] = sum - svm.linearbias_;
		//	 cuPrintf( "Score[%d] -  %f\n",tid,sum); 
		//}
	}
	
}

__global__ void showMatrixLinear(double *device){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.x;
	int j = threadIdx.x;
	cuPrintf(" show matrixLinear[%d] = %f \n",tid,device[tid]);
}

void CudaHOG::extractFeatures( unsigned char *data, int channels ){
	assert( cudaDeviceSynchronize() == cudaSuccess );
	computeGradients( hogS, data, channels );
	//printf(" blocks %d threads %d \n",blocks, threads );
	if( this->oneThreadWindow ){
		cudaSlidingWindow<<< blocks, threads>>>( hogS , svm );
	}else{
		int threads_height = (( hogS.frameH - hogS.descHeight )/ hogS.blockStride ) + 1;
		int threads_width = (( hogS.frameW - hogS.descWidth )/ hogS.blockStride ) + 1;
		dim3 blocks_windows( threads, blocks );
		dim3 threads_blocks( threads_width, threads_height );
		cudaHogThreadBlock<<< blocks_windows, threads_blocks >>>( hogS, svm );
	}
	//cudaPrintfDisplay( stdout );
}

/* ComputeGradientsInCuda */
__global__ void cudaComputeGradients( DeviceHOG hog, unsigned char* gray, int width, int height, float *mag,float *grad ){
	bool fullCircle = false;
	int p1 = 0, p2 = 0, p3 = 0, p4 = 0;
	int hor = 0, ver = 0;
    float curGrad = 0.0, curMag = 0.0;
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(  ( blockIdx.x > 1 && blockIdx.x < (height - 1) ) 
		&& ( threadIdx.x > 1 && threadIdx.x < (width - 1) ) ){
		if( tid < width * height ){
			p1 = (int) gray[ blockIdx.x*blockDim.x + threadIdx.x+1 ];
			p2 = (int) gray[ blockIdx.x*blockDim.x + threadIdx.x-1 ];
			p3 = (int) gray[ (blockIdx.x-1)*blockDim.x + threadIdx.x ];
			p4 = (int) gray[ (blockIdx.x+1)*blockDim.x + threadIdx.x ];		

			hor = p1 - p2;
			ver = p3 - p4;
			
			curMag = (float) sqrt( (double)( hor * hor ) + ( ver * ver ) );
			mag[tid] = curMag;
			// make sure we don't divide by zero when calculating the gradient orientation
			if( curMag > 0.0 ) {
				curGrad = ( (float) ( (float) 180 * acos( (float) hor / (float) curMag ) ) / (float) M_PI );
				if( !fullCircle )
					curGrad = float( (int) curGrad % 180 ); //if unsigned, then range it over 0-180 (pedestrian)
				grad[tid]=curGrad;
			}else {
				grad[tid]=0;
			}
			int angle = int( curGrad / hog.histBinSpace );
			hog.angle[tid] = ( angle >= hog.numHistBins ? hog.numHistBins - 1 : angle );
		}
	}
}

__global__ void cudaBGR2Gray_(unsigned char* bgr, unsigned char* gray,int width, int height){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid < width * height ){
		gray[tid] = ( 2989 * bgr[tid*3+2] + 5870 * bgr[tid*3+1] + 1140 * bgr[tid*3+0] ) / 10000; 
	}
}

void CudaHOG::computeGradients(DeviceHOG hog, unsigned char* data, int channels){
	unsigned char *bgr;
	if( channels == 1 ){ 
		assert( cudaMemcpy( gray, data, this->imageHeight * imageWidth, cudaMemcpyHostToDevice ) == cudaSuccess );
	}else{
		assert( cudaMalloc( (void**)&bgr, this->imageHeight * imageWidth * channels ) == cudaSuccess );
		assert( cudaMemcpy( bgr, data, this->imageHeight * imageWidth * channels, cudaMemcpyHostToDevice ) == cudaSuccess );
		cudaBGR2Gray_<<< imageHeight, imageWidth >>>( bgr, gray, this->imageWidth, this->imageHeight );
		assert( cudaFree( bgr ) == cudaSuccess );
	}
	cudaComputeGradients<<< imageHeight, imageWidth >>>(hog, gray, this->imageWidth, this->imageHeight, this->device_mag, this->device_grad );
}

float* CudaHOG::getMagnitudeN(){
	float *host_magnitude = new float[ this->imageWidth * this->imageHeight ];
	assert( cudaMemcpy( host_magnitude, this->device_mag, this->imageWidth * this->imageHeight * sizeof( float ), cudaMemcpyDeviceToHost ) == cudaSuccess );
	return host_magnitude;
}

float* CudaHOG::getGradientN(){
	float *host_gradient = new float[ this->imageWidth * this->imageHeight ];
	assert( cudaMemcpy( host_gradient, this->device_grad, this->imageWidth * this->imageHeight * sizeof( float ), cudaMemcpyDeviceToHost ) == cudaSuccess );
	return host_gradient;
}

void CudaHOG::getFoundLocations( vector<CudaPoint> &founds ){
	float **hogVector = new float*[ getWindowsCount() ];
	for( int i = 0; i < getWindowsCount(); i++ ){
		hogVector[i] = new float[ getHOGVectorSize() ];

	}
	getHOGVectorN( hogVector );
	int px=0,py=0,count=0;			
	for( int i = 0; i < getBlocksCount(); i++){
		for( int j = 0; j < getThreadsPerBlockCount(); j++ ){
			px = j * STEP_WIDTH;
			py = i * STEP_HEIGHT;
			float scoreHOGSVM = classifier->run( hogVector[count++], getHOGVectorSize(), LIGHTSVM );					 
			if( scoreHOGSVM > 0 ){
				founds.push_back(CudaPoint(px,py));
			}
		}
	}
	delete[] hogVector;
}

void CudaHOG::getHOGVectorN(float **matrix){
	matrixCpyDeviceToHost( matrix, device_hog,hogS.pitch_hog, hogSize, getWindowsCount());
}

double *CudaHOG::getScoresN(){
	double *scores = new double[ getWindowsCount() ];
	assert( cudaMemcpy( scores, svm.scores, getWindowsCount() * sizeof(double), cudaMemcpyDeviceToHost ) == cudaSuccess );
	return scores;
}