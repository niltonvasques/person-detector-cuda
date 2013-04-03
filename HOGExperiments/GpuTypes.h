#ifndef GPU_TYPES_H
#define GPU_TYPES_H

/*types declaration*/

#ifndef uchar
#define uchar unsigned char
#endif

namespace Cuda{

	struct Size{
		Size() : width( 0 ), height( 0 )
		{	}
		Size( int width_, int height_ ) : width( width_), height( height_)
		{	}

		__device__ Size( int width_, int height_, bool device ) : width( width_ ), height( height_ )
		{	}

		int width;
		int height;
	};
}
#endif