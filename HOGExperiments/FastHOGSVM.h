#ifndef __HOG_SVM_SLIDER__
#define __HOG_SVM_SLIDER__

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

__host__ void InitSVM(float svmBias, float* svmWeights, int svmWeightsCount);
__host__ void CloseSVM();

__global__ void linearSVMEvaluation(float1* svmScores, float svmBias,
									float1* blockHistograms, int noHistogramBins,
									int windowSizeX, int windowSizeY, int hogBlockCountX, int hogBlockCountY,
									int cellSizeX, int cellSizeY,
									int numberOfBlockPerWindowX, int numberOfBlockPerWindowY,
									int blockSizeX, int blockSizeY,
									int alignedBlockDimX,
									int scaleId, int scaleCount,
									int hNumberOfWindowsX, int hNumberOfWindowsY,
									int width, int height);

__host__ void ResetSVMScores(FastHOG_::FastHOG *fHOG, float1* svmScores);
__host__ void LinearSVMEvaluation(FastHOG_::FastHOG *fHOG, float1* svmScores, float1* blockHistograms, int noHistogramBins,
								  int windowSizeX, int windowSizeY,
								  int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
								  int hogBlockCountX, int hogBlockCountY,
								  int scaleId, int width, int height);

#endif
