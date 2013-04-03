#ifndef GPU_IMAGE_H
#define GPU_IMAGE_H

#include "GpuTypes.h"

namespace Cuda{
	namespace Device{
		template<typename _Tp> class GpuImage{
		public:
			_Tp *data;
			_Tp *host_data;
			int channels;
			Size size;
			size_t length_;
			__host__ GpuImage( )
			{

			}

			__host__ GpuImage( Size size_, int channels_ ) : size(size_), channels(channels_)
			{

				length_ = size.width * size.height * channels * sizeof( _Tp );
				cudaMalloc( &data, length_ );
				host_data = NULL;
			}

			__device__ GpuImage( unsigned char* data_, Size size_, int channels_ ) : size(size_), channels(channels_)
			{
				length_ = size.width * size.height * channels * sizeof( _Tp );
				cudaMalloc( &data, length_ );
				cudaMemcpy( data, data_, length_, cudaMemcpyHostToDevice );
				host_data = NULL;
			}

			__host__ void upload( _Tp* data_ ) 
			{ 
				host_data = data_;
				cudaMemcpy( data, data_, length_, cudaMemcpyHostToDevice );
			}

			__host__ void convertToFloat( GpuImage<float> m, double beta ){
				convert_to_f( *this, m, beta );
			}

			__host__ void convertToChar( GpuImage<uchar> m, double beta ){
				convert_float_to_char( *this, m, beta );
			}

			__host__ __device__ _Tp* ptr( int pos ) 
			{ 
				return &(data[pos]);	
			}

			__host__ __device__ size_t length() 
			{ 
				return length_;	
			}

			__host__ _Tp* host_ptr( int pos ){
				return &( host_data[ pos ] );
			}

			__host__ void download(){
				if( !host_data ){
					host_data = (_Tp*) malloc( length_ );
				}
				cudaMemcpy( host_data, data, length_, cudaMemcpyDeviceToHost );
			}

			//__device__ GpuImage( const Cuda::Image &image ) : size( image.size ), channels( image.channels )
			//{	
			//	size_t length = size.width * size.height * channels * sizeof( unsigned char );
			//	cudaMalloc( &data, length );
			//	cudaMemcpy( data, image.data, length, cudaMemcpyHostToDevice );
			//}
		
			__host__ void release( bool deallocate_host = true ){
				cudaFree( data );
				if( host_data && deallocate_host ) free( host_data );
			}
		};
	}

	__host__ void gpuResizeBilinearGrayF( Device::GpuImage<float> src, Device::GpuImage<float> dst );

}

__host__ void gpuColorBGR2GrayF( Cuda::Device::GpuImage<float> src, Cuda::Device::GpuImage<float> dst );
__host__ void convert_to_f( Cuda::Device::GpuImage<uchar> src,  Cuda::Device::GpuImage<float> dst, double beta );
__host__ void convert_float_to_char( Cuda::Device::GpuImage<float> src,  Cuda::Device::GpuImage<uchar> dst, double beta );

#endif