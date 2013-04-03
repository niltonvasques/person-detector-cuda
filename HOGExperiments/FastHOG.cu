#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include "FastHOG.h"
#include "FastHOGSVM.h"
#include "FastHOGScale.h"
#include "FastHOGPadding.h"
#include "FastHOGUtils.h"

using namespace FastHOG_;

__host__ void FastHOG::InitHOG(int width, int height,
					  int _avSizeX, int _avSizeY,
					  int _marginX, int _marginY,
					  int cellSizeX, int cellSizeY,
					  int blockSizeX, int blockSizeY,
					  int windowSizeX, int windowSizeY,
					  int noOfHistogramBins, float wtscale,
					  float svmBias, float* svmWeights, int svmWeightsCount,
					  bool useGrayscale)
{
	//cudaSetDevice( cutGetMaxGflopsDeviceId() );
	cudaFree( 0 );

	int i;
	int toaddxx = 0, toaddxy = 0, toaddyx = 0, toaddyy = 0;

	hWidth = width; hHeight = height;
	avSizeX = _avSizeX; avSizeY = _avSizeY; marginX = _marginX; marginY = _marginY;

	if (avSizeX) { toaddxx = hWidth * marginX / avSizeX; toaddxy = hHeight * marginY / avSizeX; }
	if (avSizeY) { toaddyx = hWidth * marginX / avSizeY; toaddyy = hHeight * marginY / avSizeY; }

	hPaddingSizeX = MAX(toaddxx, toaddyx); hPaddingSizeY = MAX(toaddxy, toaddyy);

	hPaddedWidth = hWidth + hPaddingSizeX*2;
	hPaddedHeight = hHeight + hPaddingSizeY*2;

	hUseGrayscale = useGrayscale;

	hNoHistogramBins = noOfHistogramBins;
	hCellSizeX = cellSizeX; hCellSizeY = cellSizeY; hBlockSizeX = blockSizeX; hBlockSizeY = blockSizeY;
	hWindowSizeX = windowSizeX; hWindowSizeY = windowSizeY;

	hNoOfCellsX = hPaddedWidth / cellSizeX;
	hNoOfCellsY = hPaddedHeight / cellSizeY;

	hNoOfBlocksX = hNoOfCellsX - blockSizeX + 1;
	hNoOfBlocksY = hNoOfCellsY - blockSizeY + 1;

	hNumberOfBlockPerWindowX = (windowSizeX - cellSizeX * blockSizeX) / cellSizeX + 1;
	hNumberOfBlockPerWindowY = (windowSizeY - cellSizeY * blockSizeY) / cellSizeY + 1;

	hNumberOfWindowsX = 0;
	for (i=0; i<hNumberOfBlockPerWindowX; i++) hNumberOfWindowsX += (hNoOfBlocksX-i)/hNumberOfBlockPerWindowX;

	hNumberOfWindowsY = 0;
	for (i=0; i<hNumberOfBlockPerWindowY; i++) hNumberOfWindowsY += (hNoOfBlocksY-i)/hNumberOfBlockPerWindowY;

	scaleRatio = 1.05f;
	startScale = 1.0f;
	endScale = MIN(hPaddedWidth / (float) hWindowSizeX, hPaddedHeight / (float) hWindowSizeY);
	scaleCount = (int)floor(logf(endScale/startScale)/logf(scaleRatio)) + 1;

	assert( cudaMalloc((void**) &paddedRegisteredImage, sizeof(float4) * hPaddedWidth * hPaddedHeight) == cudaSuccess );

	if (useGrayscale)
		assert( cudaMalloc((void**) &resizedPaddedImageF1, sizeof(float1) * hPaddedWidth * hPaddedHeight) == cudaSuccess );
	else
		assert( cudaMalloc((void**) &resizedPaddedImageF4, sizeof(float4) * hPaddedWidth * hPaddedHeight) == cudaSuccess );

	assert( cudaMalloc((void**) &colorGradientsF2, sizeof(float2) * hPaddedWidth * hPaddedHeight) == cudaSuccess );
	assert( cudaMalloc((void**) &blockHistograms, sizeof(float1) * hNoOfBlocksX * hNoOfBlocksY * cellSizeX * cellSizeY * hNoHistogramBins) == cudaSuccess );
	assert( cudaMalloc((void**) &cellHistograms, sizeof(float1) * hNoOfCellsX * hNoOfCellsY * hNoHistogramBins) == cudaSuccess );

	assert( cudaMalloc((void**) &svmScores, sizeof(float1) * hNumberOfWindowsX * hNumberOfWindowsY * scaleCount) == cudaSuccess );

	InitConvolution(hPaddedWidth, hPaddedHeight, useGrayscale);
	InitHistograms(cellSizeX, cellSizeY, blockSizeX, blockSizeY, noOfHistogramBins, wtscale);
	InitSVM(svmBias, svmWeights, svmWeightsCount);
	InitScale(hPaddedWidth, hPaddedHeight);
	InitPadding(hPaddedWidth, hPaddedHeight);

	rPaddedWidth = hPaddedWidth;
	rPaddedHeight = hPaddedHeight;

	if (useGrayscale)
		assert( cudaMalloc((void**) &outputTest1, sizeof(uchar1) * hPaddedWidth * hPaddedHeight) == cudaSuccess );
	else
		assert( cudaMalloc((void**) &outputTest4, sizeof(uchar4) * hPaddedWidth * hPaddedHeight) == cudaSuccess );

	assert( cudaMallocHost((void**)&hResult, sizeof(float) * hNumberOfWindowsX * hNumberOfWindowsY * scaleCount) == cudaSuccess );
}

__host__ void FastHOG::CloseHOG()
{
	assert( cudaFree(paddedRegisteredImage) == cudaSuccess );

	if (hUseGrayscale)
		assert( cudaFree(resizedPaddedImageF1) == cudaSuccess );
	else
		assert( cudaFree(resizedPaddedImageF4) == cudaSuccess );

	assert( cudaFree(colorGradientsF2) == cudaSuccess );
	assert( cudaFree(blockHistograms) == cudaSuccess );
	assert( cudaFree(cellHistograms) == cudaSuccess );

	assert( cudaFree(svmScores) == cudaSuccess );

	CloseConvolution();
	CloseHistogram();
	CloseSVM();
	CloseScale();
	ClosePadding();

	if (hUseGrayscale)
		assert( cudaFree(outputTest1) == cudaSuccess );
	else
		assert( cudaFree(outputTest4) == cudaSuccess );

	assert( cudaFreeHost(hResult) == cudaSuccess );

	assert( cudaThreadExit() == cudaSuccess );

	cutilSafeCall( cudaDeviceReset() );
}

__host__ void FastHOG::BeginProcess(unsigned char* hostImage,
		int _minx, int _miny, int _maxx, int _maxy, float minScale, float maxScale, float scaleRatio_)
{
	minX = _minx, minY = _miny, maxX = _maxx, maxY = _maxy;

	if (minY == -1 && minY == -1 && maxX == -1 && maxY == -1)
	{
		minX = 0;
		minY = 0;
		maxX = imageWidth;
		maxY = imageHeight;
	}

	int i;
	PadHostImage(this, (uchar4*)hostImage, paddedRegisteredImage, minX, minY, maxX, maxY);

	rPaddedWidth = hPaddedWidth; rPaddedHeight = hPaddedHeight;
	float scale = scaleRatio_;
	scaleRatio = 100 / (scale * 100);
	startScale = (minScale < 0.0f) ? 1.0f : minScale;
	endScale = (maxScale < 0.0f) ? MIN(hPaddedWidth / (float) hWindowSizeX, hPaddedHeight / (float) hWindowSizeY) : maxScale;
	scaleCount = (int)floor(logf(endScale/startScale)/logf(scaleRatio)) + 1;

	float currentScale = startScale;

	ResetSVMScores(this, svmScores);

	nwindows = 0;

	for (i=0; i<scaleCount; i++)
	{
		DownscaleImage( this, 0, scaleCount, i, currentScale, hUseGrayscale, paddedRegisteredImage, resizedPaddedImageF1, resizedPaddedImageF4);

		SetConvolutionSize(rPaddedWidth, rPaddedHeight);

		if(hUseGrayscale) ComputeColorGradients1to2(resizedPaddedImageF1, colorGradientsF2);
		else ComputeColorGradients4to2(resizedPaddedImageF4, colorGradientsF2);

		ComputeBlockHistogramsWithGauss(colorGradientsF2, blockHistograms, hNoHistogramBins,
			hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY, hWindowSizeX, hWindowSizeY,  rPaddedWidth, rPaddedHeight);

		NormalizeBlockHistograms(blockHistograms, hNoHistogramBins, hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY, rPaddedWidth, rPaddedHeight);

		LinearSVMEvaluation(this, svmScores, blockHistograms, hNoHistogramBins, hWindowSizeX, hWindowSizeY, hCellSizeX, hCellSizeY,
			hBlockSizeX, hBlockSizeY, rNoOfBlocksX, rNoOfBlocksY, i, rPaddedWidth, rPaddedHeight);

		const int v = 1 + (rPaddedHeight - hWindowSizeY) / 8;
		const int h = 1 + (rPaddedWidth - hWindowSizeX) / 8;
		nwindows += h*v;

		currentScale *= scaleRatio;
	}
}

//__host__ float* EndHOGProcessing()
//{
//	cudaThreadSynchronize();
//	cutilSafeCall(cudaMemcpy(hResult, svmScores, sizeof(float) * scaleCount * hNumberOfWindowsX * hNumberOfWindowsY, cudaMemcpyDeviceToHost));
//
//	for( int i =0; i < scaleCount * hNumberOfWindowsX * hNumberOfWindowsY; printf("Results Score %f\n",hResult[i++] ) );
//	return hResult;
//}

void FastHOG::FinalizeHOG()
{
	delete nmsProcessor;

	CloseHOG();
}

__host__ void FastHOG::EndProcess()
{
	cppResult = hResult;
	cudaThreadSynchronize();
	cutilSafeCall(cudaMemcpy(cppResult, svmScores, sizeof(float) * scaleCount * hNumberOfWindowsX * hNumberOfWindowsY, cudaMemcpyDeviceToHost));

	ComputeFormattedResults();

	nmsResults = nmsProcessor->ComputeNMSResults(formattedResults, formattedResultsCount, &nmsResultsAvailable, &nmsResultsCount,
		hWindowSizeX, hWindowSizeY);
}

void FastHOG::ComputeFormattedResults()
{
	int i, j, k, resultId;
	int leftoverX, leftoverY, currentWidth, currentHeight, rNumberOfWindowsX, rNumberOfWindowsY;

	resultId = 0;
	formattedResultsCount = 0;

	float* currentScaleWOffset;
	float currentScale = startScale;

	for (i=0; i<scaleCount; i++)
	{
		currentScaleWOffset = cppResult + i * hNumberOfWindowsX * hNumberOfWindowsY;

		for (j = 0; j < hNumberOfWindowsY; j++)
		{
			for (k = 0; k < hNumberOfWindowsX; k++)
			{
				float score = currentScaleWOffset[k + j * hNumberOfWindowsX];
				if (score > 0)
					formattedResultsCount++;
			}
		}
	}

	if (formattedResultsAvailable) delete formattedResults;
	formattedResults = new FastHOGResult[formattedResultsCount];

	for (i=0; i<scaleCount; i++)
	{
		currentScaleWOffset = cppResult + i * hNumberOfWindowsX * hNumberOfWindowsY;

		for (j=0; j<hNumberOfWindowsY; j++)
		{
			for (k=0; k<hNumberOfWindowsX; k++)
			{
				float score = currentScaleWOffset[k + j * hNumberOfWindowsX];
				if (score > 0)
				{
					FastHOGResult hogResult;

					currentWidth = iDivUpF(hPaddedWidth, currentScale);
					currentHeight = iDivUpF(hPaddedHeight, currentScale);

					rNumberOfWindowsX = (currentWidth - hWindowSizeX) / hCellSizeX + 1;
					rNumberOfWindowsY = (currentHeight - hWindowSizeY) / hCellSizeY + 1;

					leftoverX = (currentWidth - hWindowSizeX - hCellSizeX * (rNumberOfWindowsX - 1)) / 2;
					leftoverY = (currentHeight - hWindowSizeY - hCellSizeY * (rNumberOfWindowsY - 1)) / 2;

					hogResult.origX = k * hCellSizeX + leftoverX;
					hogResult.origY = j * hCellSizeY + leftoverY;

					hogResult.width = (int)floorf((float)hWindowSizeX * currentScale);
					hogResult.height = (int)floorf((float)hWindowSizeY * currentScale);

					hogResult.x = (int)ceilf(currentScale * (hogResult.origX + hWindowSizeX / 2) - (float) hWindowSizeX * currentScale / 2) - hPaddingSizeX + minX;
					hogResult.y = (int)ceilf(currentScale * (hogResult.origY + hWindowSizeY / 2) - (float) hWindowSizeY * currentScale / 2) - hPaddingSizeY + minY;

					hogResult.scale = currentScale;
					hogResult.score = score;

					formattedResults[resultId] = hogResult;
					resultId++;
				}
			}
		}

		currentScale = currentScale * scaleRatio;
	}
}

void FastHOG::SaveResultsToDisk(char* fileName)
{
	FILE* f; 
#ifdef _WIN32
	fopen_s(&f, fileName, "w+");
#else
	f = fopen(fileName, "w+");
#endif
	fprintf(f, "%d\n", formattedResultsCount);
	for (int i=0; i<formattedResultsCount; i++)
	{
		fprintf(f, "%f %f %d %d %d %d %d %d\n",
			formattedResults[i].scale, formattedResults[i].score,
			formattedResults[i].width, formattedResults[i].height,
			formattedResults[i].x, formattedResults[i].y,
			formattedResults[i].origX, formattedResults[i].origY);
	}
	fclose(f);
}
