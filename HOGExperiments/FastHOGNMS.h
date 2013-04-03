#ifndef __HOG_NMS__
#define __HOG_NMS__

#include "FastHOGPoint3.h"
#include "FastHOGResult.h"

namespace FastHOG_
{
	class FastHOGNMS
	{
	private:
		FastHOGPoint3 *at, *ms, *tomode, *nmsToMode;

		FastHOGResult *nmsResults;

		float* wt;

		float center, scale;
		float nonmaxSigma[3];
		float nsigma[3];
		float modeEpsilon;
		float epsFinalDist;

		int maxIterations;

		bool isAllocated;

		float sigmoid(float score) { return (score > center) ? scale * (score - center) : 0.0f; }
		void nvalue(FastHOGPoint3* ms, FastHOGPoint3* at, float* wt, int length);
		void nvalue(FastHOGPoint3* ms, FastHOGPoint3* msnext, FastHOGPoint3* at, float* wt, int length);
		void fvalue(FastHOGPoint3* modes, FastHOGResult* results, int lengthModes, FastHOGPoint3* at, float* wt, int length);
		void shiftToMode(FastHOGPoint3* ms, FastHOGPoint3* at, float* wt, FastHOGPoint3 *tomode, int length);
		float distqt(FastHOGPoint3 *p1, FastHOGPoint3 *p2);

	public:
		FastHOGResult* ComputeNMSResults(FastHOGResult* formattedResults, int formattedResultsCount, bool *nmsResultsAvailable, int *nmsResultsCount,
			int hWindowSizeX, int hWindowSizeY);

		FastHOGNMS();
		~FastHOGNMS(void);
	};
}
#endif
