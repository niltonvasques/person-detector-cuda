#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "image.h"
#include "HOGCudaAlloc.h"
#include "gpu_utils.h"
//#include "cuPrintf.cu"

#define fmin( a, b ) ( (a)<(b)? (a):(b) )
template<typename T> inline bool isnan(T value){
	return value != value;
}

__global__ void cudaPowerLawGammaCorrection2(unsigned char* pixels, int width, int height, int channels,double gamma){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if(tid < width * height){
			pixels[tid*channels+0]=255*pow(((double)pixels[tid*channels+0] / 255.0 ),(double)1.0/gamma);
			pixels[tid*channels+1]=255*pow(((double)pixels[tid*channels+1] / 255.0 ),(double)1.0/gamma);
			pixels[tid*channels+2]=255*pow(((double)pixels[tid*channels+2] / 255.0 ),(double)1.0/gamma);
	}
}

/* ComputeGradientsInCuda */
__global__ void cudaNativeComputeGradients( unsigned char* gray, int width, int height, float *mag,float *grad ){
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
		}
	}
}

__global__ void cudaNativeBGR2Gray(unsigned char* bgr, unsigned char* gray,int width, int height){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid < width * height ){
		gray[tid] = ( 2989 * bgr[tid*3+2] + 5870 * bgr[tid*3+1] + 1140 * bgr[tid*3+0] ) / 10000; 
	}
}

__global__ void cudaNativeComputeDescriptor( int bx, int by, float* desc, int descHeight, int descWidth, float *mag, float *grad,
	int frameW, int numHistBins, int histBinSpace, int blockHeight, int blockWidth,int cellWidth) {
    
    float curGrad = 0.0, curMag = 0.0, gWeight = 0.0, 
            cellWeight = 0.0, binWeight = 0.0,
            dist = 0.0;
    int i = 0, iBin = 0,
         stepx = 0, stepy = 0,
         mx = 0, my = 0,
         cx = 0, cy = 0,
         dx = 0, dy = 0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if( tid < (descHeight * descWidth) ){
	//for(int y = 0; y < descHeight; y++){
	//	for(int x = 0; x < descWidth; x++){
			int x = threadIdx.x;
			int y = blockIdx.x;
			curGrad		= grad[ ((by+y)*frameW) + bx+x ]; 
			curMag		= mag[ ((by+y)*frameW) + bx+x ]; 
			i = int( curGrad / histBinSpace );
			if( i >= numHistBins ) i = numHistBins - 1;
	    
			mx = int( descWidth >> 1 );
			my = int( descHeight >> 1 );

			//gWeight = gaussian( float(x), float(y), float(mx), float(my), float(mx) );
			gWeight = GAUSSIAN( float(x), float(y), float(mx), float(my), float(mx) );
			//cuPrintf("x: %d  y: %d  mx: %d my %d  Gaussian Weight: %f \n",x,y,mx,my,gWeight);
			stepx = mx;
			stepy = my;
			for( int iy = 0; iy < blockHeight; iy++ ) {
				for( int ix = 0; ix < blockWidth; ix++ ) {
					cx = stepx + stepx * ix;
					cy = stepy + stepy * iy;
					dx = x - cx;
					dy = y - cy;
					dist = sqrt( (float) ( dx * dx ) + ( dy * dy ) );    
					//cell weighting
					cellWeight = 1.0 - fmin( (float) ( (float) dist / (float) cellWidth ), (float) 1.0 );
					//cuPrintf(" cellWeight %f \n",cellWeight);			    		    
					// histogram bin weighting
					iBin = 0; binWeight = 0;
					int halfSize = int( histBinSpace >> 1 );
					float value = curGrad - histBinSpace * i;
					if( value > halfSize ) { // range: (halfSize, binsize]
						iBin = ( i + 1 ) % numHistBins;
						binWeight = 1.0 - ( ( value - halfSize ) / histBinSpace );
					} else { // range: [0, halfsize]
						iBin = ( i - 1 ) % numHistBins;
						if( iBin < 0 ) iBin += numHistBins;
						binWeight = ( ( value + halfSize ) / histBinSpace );
					}
					
					int offset1 = (( (i) * ( (blockWidth) * (blockHeight) ) + (iy) * (blockWidth) + (ix)) );
					int offset2 = (( (iBin) * ( (blockWidth) * (blockHeight) ) + (iy) * (blockWidth) + (ix)) );
					atomicAdd((float*)&desc[offset1], (float)(binWeight * cellWeight * gWeight * curMag));
					atomicAdd((float*)&desc[offset2], (float)((1.0 - binWeight) * cellWeight * (gWeight) * curMag) );
				}
			}
		//}
	}
}

/* Descobrir como paralelizar este método */
void cudaComputeDescriptors(int bx, int by, float** device_desc, int descHeight, int descWidth, float** device_mag, float** device_grad,
	int frameW, int numHistBins, int histBinSpace, int blockHeight, int blockWidth,int cellWidth){
	size_t size = sizeof(float) * numHistBins*blockWidth*blockHeight;
	cudaMemset( *device_desc, 0, size ); //Reset  Descriptor

	cudaNativeComputeDescriptor<<< descHeight, descWidth >>>( bx, by, *device_desc, descHeight, descWidth, *device_mag, *device_grad,
	frameW, numHistBins, histBinSpace, blockHeight, blockWidth, cellWidth);
	
	//cudaPrintfDisplay(stdout, true);
}

void cudaComputeGradients(unsigned char* pixels, int width, int height,float **mag,float **grad ,int channels){
	int size = width*height*channels;
	unsigned char *pixels_device,*gray_device;

	cudaMalloc( &pixels_device, size );
	cudaMemcpy( pixels_device, pixels, size, cudaMemcpyHostToDevice );

	if(channels >= 3){
		//Timer t;
		//t.start();
		cudaMalloc( &gray_device, width * height );
		cudaNativeBGR2Gray<<< width,height >>>( pixels_device, gray_device, width, height );
		cudaNativeComputeGradients<<< height,width >>>( gray_device, width, height, *mag, *grad );
		//t.stop();
		//t.check("cudaNativeComputeGradients ");
		cudaFree( gray_device );
	}else{
		cudaNativeComputeGradients<<< width,height >>>( pixels_device, width, height, *mag, *grad );
	}

	//cudaMemcpy( mag, mag_device, width*height*sizeof( float ), cudaMemcpyDeviceToHost );
	//cudaMemcpy( grad, grad_device, width*height*sizeof( float ), cudaMemcpyDeviceToHost );
	
	cudaFree( pixels_device );
	//cudaFree( mag_device );
	//cudaFree( grad_device );
}

__global__ void cudaNativeCalculateL2Hys( int numHistBins, int blockHeight, int blockWidth, float *desc, float *norm ) {
    
    float eps = 1.0;
	//int k = threadIdx.x;
	int i = blockIdx.x;
	int j = threadIdx.x;

    //compute norm
  //  for( int i = 0; i < numHistBins; i++ ){
		//for( int j = 0; j < blockHeight; j++ ){
	for( int k = 0; k < blockWidth; k++ ){
		float val = CUDA_DESCRIPTOR_ELEMENT( desc, float, k, j, i, blockWidth, blockHeight, numHistBins );
		atomicAdd(norm,val*val);
	}
	//	}
	//}
	__syncthreads();
	//Sincronize All Threads
	//L2-norm
	float norm_ = sqrt( (*norm) + eps ); 
    
	if ( !norm_ ) norm_ = 1.0;
    
    // Normalize and threshold ...
  //  for( int i = 0; i < numHistBins; i++ ){
		//for( int j = 0; j < blockHeight; j++ ){
	for( int k = 0; k < blockWidth; k++ ) {
		float *element = &CUDA_DESCRIPTOR_ELEMENT( desc, float, k, j, i, blockWidth, blockHeight, numHistBins );
		(*element) /= norm_;
		if( (*element) > 0.2 ) (*element) = 0.2;
	}
	//	}
	//}
    
    *norm = 0.0;
	__syncthreads();
  //  for( int i = 0; i < numHistBins; i++ ){
		//for( int j = 0; j < blockHeight; j++ ){
	for( int k = 0; k < blockWidth; k++ ){
		float val = CUDA_DESCRIPTOR_ELEMENT( desc, float, k, j, i, blockWidth, blockHeight, numHistBins );
		atomicAdd(norm, val*val);
	}
	//	}
	//}
    
    __syncthreads();
	norm_ = sqrt( *norm + eps );
    if ( !norm_ ) norm_ = 1.0;
    // and normalize again
  //  for( int i = 0; i < numHistBins; i++ ){
		//for( int j = 0; j < blockHeight; j++ ){
	for( int k = 0; k < blockWidth; k++ ){
		CUDA_DESCRIPTOR_ELEMENT( desc, float, k, j, i, blockWidth, blockHeight, numHistBins ) /= norm_;
	}
	//	}
	//}
    
}

void cudaCalculateL2Hys( int numHistBins, int blockHeight, int blockWidth, float ** device_desc ){
	//dim3 blockDim( blockWidth, blockHeight, numHistBins );
	float *norm;
	float norm_host = 0.0;
	cudaMalloc((void**)&norm,sizeof(float));
	cudaMemcpy( norm, &norm_host, sizeof(float), cudaMemcpyHostToDevice );
	cudaNativeCalculateL2Hys<<< numHistBins, blockHeight >>>( numHistBins, blockHeight, blockWidth, *device_desc, norm);
	cudaFree( norm );
}

__global__ void cudaNativeWriteToVector( float* device_hog, int index , int numHistBins, int blockHeight, int blockWidth, float* device_desc ) {
	float *output = &(device_hog[index]);
	float feat = 0.0;
	int i = blockIdx.x * blockDim.x * blockWidth + threadIdx.x * blockWidth;
	int b = blockIdx.x;
	int y = threadIdx.x;
	for( int x = 0; x < blockWidth; x++ ) {				
		feat = CUDA_DESCRIPTOR_ELEMENT( device_desc, float, x, y, b, blockWidth, blockHeight, numHistBins );
		if( isnan( feat ) ) {
			feat = 0;
		}
		device_hog[index+i] = feat;
		i++;
	}
}

void cudaWriteToVector( float** device_hog, int index , int numHistBins, int blockHeight, int blockWidth, float** device_desc ){
	cudaNativeWriteToVector<<< numHistBins, blockHeight >>>( *device_hog, index , numHistBins, blockHeight, blockWidth, *device_desc );
}


void cudaPowerLawGammaCorrection(unsigned char* pixelsSrc,unsigned char* pixelsDst, int width, int height, int channels,double gamma){
	int size = width*height*channels;
	unsigned char *pixels_device;
	cudaMalloc(&pixels_device,size);
	cudaMemcpy(pixels_device,pixelsSrc,size,cudaMemcpyHostToDevice);
	cudaPowerLawGammaCorrection2<<<width,height>>>(pixels_device,width,height,channels,gamma);
	cudaMemcpy(pixelsDst,pixels_device,size,cudaMemcpyDeviceToHost);
	cudaFree(pixels_device);
}

void cudaBGR2Gray(unsigned char* bgr, unsigned char *gray, int width, int height){
	int size = width*height*3;
	unsigned char *bgr_device,*gray_device;
	cudaMalloc(&bgr_device,size);
	cudaMalloc(&gray_device,width*height);
	cudaMemcpy(bgr_device,bgr,size,cudaMemcpyHostToDevice);

	cudaNativeBGR2Gray<<< width,height >>>(bgr_device,gray_device,width,height);

	cudaMemcpy(gray,gray_device,width*height,cudaMemcpyDeviceToHost);

	cudaFree(bgr_device);
	cudaFree(gray_device);
}



//void cutImage( unsigned char *data, unsigned char *output,  int px, int width, int py, int height, int imageWidth, int imageHeight ){
//	for( int i = 0; i < height; i++ ){
//		for( int j = 0; j < width; j++ ){
//			int offset = (py+i) * imageWidth + (px+j);
//			output[ i * width + j ] = data[offset];			
//		}
//	}
//}

