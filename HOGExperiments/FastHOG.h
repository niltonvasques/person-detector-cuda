#ifndef FAST_HOG_H
#define FAST_HOG_H

#include <cutil_inline.h>
#include "FastHOGNMS.h"
#include "FastHOGResult.h"

// Inicio do port do fastHOG para Windows...

namespace FastHOG_
{
	class FastHOG{

	public:
			int imageWidth, imageHeight;

			int avSizeX, avSizeY, marginX, marginY;

			int hCellSizeX, hCellSizeY;
			int hBlockSizeX, hBlockSizeY;
			int hWindowSizeX, hWindowSizeY;
			int hNoHistogramBins, rNoHistogramBins;
			int hNoOfHistogramBins;
			int hPaddedWidth, hPaddedHeight;
			int rPaddedWidth, rPaddedHeight;
			int hPaddingSizeX, hPaddingSizeY;

			int minX, minY, maxX, maxY;

			float wtScale;

			float startScale, endScale, scaleRatio;

			int svmWeightsCount;
			float svmBias, *svmWeights;

			int hWidth, hHeight;
			int hWidthROI, hHeightROI;

			float4 *paddedRegisteredImage;

			float1 *resizedPaddedImageF1;
			float4 *resizedPaddedImageF4;

			float2 *colorGradientsF2;

			float1 *blockHistograms;
			float1 *cellHistograms;

			float1 *svmScores;

			bool hUseGrayscale;

			uchar1* outputTest1;
			uchar4* outputTest4;

			float* hResult;

			int scaleCount;

			int hNoOfCellsX, hNoOfCellsY;
			int hNoOfBlocksX, hNoOfBlocksY;
			int rNoOfCellsX, rNoOfCellsY, rNoOfBlocksX, rNoOfBlocksY;
			int hNumberOfWindowsX, hNumberOfWindowsY;
			int hNumberOfBlockPerWindowX, hNumberOfBlockPerWindowY;

			bool useGrayscale;

			float* cppResult;

			FastHOGResult* formattedResults;
			FastHOGResult* nmsResults;

			bool formattedResultsAvailable;
			int formattedResultsCount;

			bool nmsResultsAvailable;
			int nmsResultsCount;

			int nwindows;

	private:
		FastHOGNMS* nmsProcessor;

	public:
		FastHOG( int imageWidth, int imageHeight, float svmBias, float* svmWeights, int svmWeightsCount){
			this->imageWidth = imageWidth;
			this->imageHeight = imageHeight;

			this->avSizeX = 48; //48
			this->avSizeY = 96; //96
			this->marginX = 4; // 4
			this->marginY = 4; // 4

			this->hCellSizeX = 8;
			this->hCellSizeY = 8;
			this->hBlockSizeX = 2;
			this->hBlockSizeY = 2;
			this->hWindowSizeX = 64;
			this->hWindowSizeY = 128;
			this->hNoOfHistogramBins = 9;

			this->svmWeightsCount = svmWeightsCount;
			this->svmBias = svmBias;
			this->svmWeights = svmWeights;

			this->wtScale = 2.0f;

			this->useGrayscale = false;

			this->formattedResultsAvailable = false;

			nmsProcessor = new FastHOGNMS();

			InitHOG(imageWidth, imageHeight, avSizeX, avSizeY, marginX, marginY, hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY,
				hWindowSizeX, hWindowSizeY, hNoOfHistogramBins, wtScale, svmBias, svmWeights, svmWeightsCount, useGrayscale);
		}

		~FastHOG(){
			FinalizeHOG();
		}

		void BeginProcess(unsigned char* hostImage, int _minx = -1, int _miny = -1, int _maxx = -1, int _maxy = -1,
						float minScale = -1.0f, float maxScale = -1.0f, float scaleRatio_ = 0.95f);
		void EndProcess();

		inline int getPositivesCount(){
			return nmsResultsCount;
		}

		inline FastHOGResult* getPositives(){
			return nmsResults;
		}

	private:
		void InitHOG(int width, int height, int _avSizeX, int _avSizeY, int _marginX, int _marginY, int cellSizeX, int cellSizeY,
						  int blockSizeX, int blockSizeY, int windowSizeX, int windowSizeY, int noOfHistogramBins, float wtscale,
						  float svmBias, float* svmWeights, int svmWeightsCount, bool useGrayscale);
		void CloseHOG();
		void FinalizeHOG();
		//void GetImage(HOGImage *imageCUDA, ImageType imageType);

		void ComputeFormattedResults();

		void SaveResultsToDisk(char* fileName);


		/*         HOG Convolution			*/
		void InitConvolution(int width, int height, bool useGrayscale);
		void SetConvolutionSize(int width, int height);
		void CloseConvolution();
		void ComputeColorGradients1to2(float1* inputImage, float2* outputImage);
		void ComputeColorGradients4to2(float4* inputImage, float2* outputImage);

		/*			HOG Histogram			*/
		void InitHistograms(int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY, int noHistogramBins, float wtscale);
		void CloseHistogram();
		void ComputeBlockHistogramsWithGauss(float2* inputImage, float1* blockHistograms, int noHistogramBins, int cellSizeX, int cellSizeY, 
												int blockSizeX, int blockSizeY, int windowSizeX, int windowSizeY, int width, int height);

		void NormalizeBlockHistograms(float1* blockHistograms, int noHistogramBins,int cellSizeX, int cellSizeY,
										int blockSizeX, int blockSizeY, int width, int height );


	};
}
#endif