#ifndef __HOG_PADDING__
#define __HOG_PADDING__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "FastHOG.h"

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <cuda_gl_interop.h>
#include <cutil_inline.h>
#include <cuda.h>
#include "FastHOGDefines.h"

__host__ void InitPadding(int hPaddedWidth, int hPaddedHeight);
__host__ void ClosePadding();

__host__ void PadHostImage(FastHOG_::FastHOG *fHOG, uchar4* registeredImage, float4 *paddedRegisteredImage,
		int minx, int miny, int maxx, int maxy);

#endif
