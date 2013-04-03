#include <iostream>
#include "HOGCuda.h"
#include "GpuImage.h"

class SaliencyCUDA{

public:
	SaliencyCUDA(){
		cudaEventCreate( &start );
		cudaEventCreate( &stop );
	}

	~SaliencyCUDA(){
		cudaEventDestroy( start );
		cudaEventDestroy( stop );
	}

	float elapsed_time;

	cudaEvent_t start, stop;

	inline Cuda::Device::GpuImage<float> Saliency( Cuda::Device::GpuImage<uchar> im, double scale_factor = 0.15 ){
		cudaEventRecord( start );

		Cuda::Size resize_to = Cuda::Size(  im.size.width * scale_factor, im.size.height * scale_factor );
		Cuda::Device::GpuImage<float> img3f( im.size, im.channels );

		im.convertToFloat( (Cuda::Device::GpuImage<float>)img3f, 1.0/255 );

		return GetSR( img3f, resize_to, false );
	}

	inline Cuda::Device::GpuImage<float> GetSR( Cuda::Device::GpuImage<float> img3f, Cuda::Size sz, bool resize_back = true ){
		Cuda::Device::GpuImage<float> img1f[2], sr1f, cmplxSrc2f, cmplxDst2f;

		img1f[1] = Cuda::Device::GpuImage<float>( img3f.size, 1 );
		img1f[0] = Cuda::Device::GpuImage<float>( sz, 1 );

		gpuColorBGR2GrayF( img3f, img1f[1] );
		img3f.release();
		Cuda::gpuResizeBilinearGrayF( img1f[1], img1f[0] ); // Usando resize Bilinear até encontrar o algoritmo do resize area.
		//cv::resize(img1f[1], img1f[0], sz, 0, 0, CV_INTER_AREA);

		Cuda::Device::GpuImage<uchar> normalized( img1f[0].size, img1f[0].channels );
		img1f[0].convertToChar( normalized, 255.f );
		//cv::equalizeHist(normalized, normalized);
		normalized.convertToFloat( img1f[0], 1.f/255.f );
		normalized.release();

		//img1f[1] = cv::Mat::zeros(sz, CV_32F);
		//cv::merge(img1f, 2, cmplxSrc2f);
		//cv::dft(cmplxSrc2f, cmplxDst2f);
		//AbsAngle(cmplxDst2f, img1f[0], img1f[1]);

		//cv::log(img1f[0], img1f[0]);
		//cv::blur(img1f[0], sr1f, cv::Size(3, 3));
		//sr1f = img1f[0] - sr1f;

		//cv::exp(sr1f, sr1f);
		//GetCmplx(sr1f, img1f[1], cmplxDst2f);
		//cv::dft(cmplxDst2f, cmplxSrc2f, CV_DXT_INVERSE | CV_DXT_SCALE);
		//cv::split(cmplxSrc2f, img1f);

		//cv::pow(img1f[0], 2, img1f[0]);
		//cv::pow(img1f[1], 2, img1f[1]);
		//img1f[0] += img1f[1];

		//cv::GaussianBlur(img1f[0], img1f[0], cv::Size(3, 3), 0);
		//cv::normalize(img1f[0], img1f[0], 0, 1, 32 /*NORM_MINMAX*/);		

		if (resize_back)
		{
			//cv::resize(img1f[0], img1f[1], img3f.size(), 0, 0, CV_INTER_CUBIC);
			cudaEventRecord( stop );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &elapsed_time, start, stop );

			return img1f[1];
		}
		else
		{
			return img1f[0];
		}
	}

};