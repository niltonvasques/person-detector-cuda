#include "FastHOGPadding.h"
#include "FastHOGUtils.h"

//extern int hWidthROI, hHeightROI;
//extern int hPaddedWidth, hPaddedHeight;
//extern int hWidth, hHeight;
//extern int hPaddingSizeX, hPaddingSizeY;

//extern int avSizeX, avSizeY, marginX, marginY;

uchar4* paddedRegisteredImageU4;

__host__ void InitPadding(int hPaddedWidth, int hPaddedHeight)
{
	cutilSafeCall(cudaMalloc((void**) &paddedRegisteredImageU4, sizeof(uchar4) * hPaddedWidth * hPaddedHeight));
}

__host__ void ClosePadding()
{
	cutilSafeCall(cudaFree(paddedRegisteredImageU4));
}

__host__ void PadHostImage(FastHOG_::FastHOG *fHOG, uchar4* registeredImage, float4 *paddedRegisteredImage,
		int minx, int miny, int maxx, int maxy)
{
	fHOG->hWidthROI = maxx - minx;
	fHOG->hHeightROI = maxy - miny;

	int toaddxx = 0, toaddxy = 0, toaddyx = 0, toaddyy = 0;

	if (fHOG->avSizeX) { toaddxx = fHOG->hWidthROI * fHOG->marginX / fHOG->avSizeX; toaddxy = fHOG->hHeightROI * fHOG->marginY / fHOG->avSizeX; }
	if (fHOG->avSizeY) { toaddyx = fHOG->hWidthROI * fHOG->marginX / fHOG->avSizeY; toaddyy = fHOG->hHeightROI * fHOG->marginY / fHOG->avSizeY; }

	fHOG->hPaddingSizeX = MAX(toaddxx, toaddyx); fHOG->hPaddingSizeY = MAX(toaddxy, toaddyy);

	fHOG->hPaddedWidth = fHOG->hWidthROI + fHOG->hPaddingSizeX*2;
	fHOG->hPaddedHeight = fHOG->hHeightROI + fHOG->hPaddingSizeY*2;

	cutilSafeCall(cudaMemset(paddedRegisteredImageU4, 0, sizeof(uchar4) * fHOG->hPaddedWidth * fHOG->hPaddedHeight));

	cutilSafeCall(cudaMemcpy2D(paddedRegisteredImageU4 + fHOG->hPaddingSizeX + fHOG->hPaddingSizeY * fHOG->hPaddedWidth,
			fHOG->hPaddedWidth * sizeof(uchar4), registeredImage + minx + miny * fHOG->hWidth,
			fHOG->hWidth * sizeof(uchar4), fHOG->hWidthROI * sizeof(uchar4),
			fHOG->hHeightROI, cudaMemcpyHostToDevice));

	Uchar4ToFloat4(paddedRegisteredImageU4, paddedRegisteredImage, fHOG->hPaddedWidth, fHOG->hPaddedHeight);
}
