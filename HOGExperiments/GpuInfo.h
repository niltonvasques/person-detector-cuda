#ifndef GPU_INFO_H
#define GPU_INFO_H

namespace Cuda{

	class GpuInfo{
		static GpuInfo *instance;

	private:
		int maxThreads;

		GpuInfo()
		{ 
				cudaDeviceProp prop;
				int dev;
				cudaGetDevice( &dev );
				cudaGetDeviceProperties( &prop, dev );
				maxThreads = prop.maxThreadsPerBlock;
		}

	public:
		static GpuInfo *getInstance();
		static void destroyInstance();

		inline public int getMaxThreads(){
			return maxThreads;
		}

	};

	GpuInfo *GpuInfo::instance = NULL;

	GpuInfo *GpuInfo::getInstance()	{
		if( !instance ) GpuInfo::instance = new GpuInfo();
		return instance;
	}

	void GpuInfo::destroyInstance(){
		delete instance;
		instance = NULL;
	}

}

#endif