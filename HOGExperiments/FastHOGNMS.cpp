#include "FastHOGNMS.h"

#include <math.h>

using namespace FastHOG_;

FastHOGNMS::FastHOGNMS()
{
	center = 0.0f; scale = 1.0f;
	nonmaxSigma[0] = 8.0f; nonmaxSigma[1] = 16.0f; nonmaxSigma[2] = 1.3f;
	maxIterations = 100;
	modeEpsilon = (float)1e-5;
	epsFinalDist = 1.0f;

	nsigma[0] = nonmaxSigma[0]; nsigma[1] = nonmaxSigma[1]; nsigma[2] = logf(nonmaxSigma[2]);

	isAllocated = false;
}

FastHOGNMS::~FastHOGNMS()
{
	if (isAllocated)
	{
		delete tomode;
		delete wt;
		delete ms;
		delete at;
		delete nmsResults;
		delete nmsToMode;
	}
}

void FastHOGNMS::nvalue(FastHOGPoint3* ms, FastHOGPoint3* at, float* wt, int length)
{
	int i, j;
	float dotxmr, w;
	FastHOGPoint3 x, r, ns, numer, denum;

	for (i=0; i<length; i++)
	{
		numer.x = 0; numer.y = 0; numer.z = 0;
		denum.x = 0; denum.y = 0; denum.z = 0;

		for (j=0; j<length; j++)
		{
			ns.x = nsigma[0] * expf(at[j].z); ns.y =  nsigma[1] * expf(at[j].z); ns.z = nsigma[2];
			x.x = at[j].x / ns.x; x.y = at[j].y / ns.y; x.z = at[j].z / ns.z;
			r.x = at[i].x / ns.x; r.y = at[i].y / ns.y; r.z = at[i].z / ns.z;

			dotxmr = (x.x - r.x) * (x.x - r.x) + (x.y - r.y) * (x.y - r.y) + (x.z - r.z) * (x.z - r.z);
			w = wt[j] * expf(-dotxmr/2.0f)/sqrtf(ns.x * ns.y * ns.z);

			numer.x += w * x.x; numer.y += w * x.y; numer.z += w * x.z;
			denum.x += w / ns.x; denum.y += w / ns.y; denum.z += w / ns.z;
		}

		ms[i].x = numer.x / denum.x; ms[i].y = numer.y / denum.y; ms[i].z = numer.z / denum.z;
	}
}

void FastHOGNMS::nvalue(FastHOGPoint3 *ms, FastHOGPoint3* msnext, FastHOGPoint3* at, float* wt, int length)
{
	int j;
	float dotxmr, w;
	FastHOGPoint3 x, r, ns, numer, denum, toReturn;

	for (j=0; j<length; j++)
	{
		ns.x = nsigma[0] * expf(at[j].z); ns.y =  nsigma[1] * expf(at[j].z); ns.z = nsigma[2];
		x.x = at[j].x / ns.x; x.y = at[j].y / ns.y; x.z = at[j].z / ns.z;
		r.x = ms->x / ns.x; r.y = ms->y / ns.y; r.z = ms->z / ns.z;

		dotxmr = (x.x - r.x) * (x.x - r.x) + (x.y - r.y) * (x.y - r.y) + (x.z - r.z) * (x.z - r.z);
		w = wt[j] * expf(-dotxmr/2.0f)/sqrtf(ns.x * ns.y * ns.z);

		numer.x += w * x.x; numer.y += w * x.y; numer.z += w * x.z;
		denum.x += w / ns.x; denum.y += w / ns.y; denum.z += w / ns.z;
	}

	msnext->x = numer.x / denum.x; msnext->y = numer.y / denum.y; msnext->z = numer.z / denum.z;
}

void FastHOGNMS::fvalue(FastHOGPoint3* modes, FastHOGResult* results, int lengthModes, FastHOGPoint3* at, float* wt, int length)
{
	int i, j;
	float no, dotxx;
	FastHOGPoint3 x, ns;
	for (i=0; i<lengthModes; i++)
	{
		no = 0;
		for (j=0; j<length; j++)
		{
			ns.x = nsigma[0] * expf(at[j].z); ns.y =  nsigma[1] * expf(at[j].z); ns.z = nsigma[2];
			x.x = (at[j].x - modes[i].x) / ns.x;
			x.y = (at[j].y - modes[i].y) / ns.y;
			x.z = (at[j].z - modes[i].z) / ns.z;

			dotxx = x.x * x.x + x.y * x.y + x.z * x.z;

			no += wt[j] * expf(-dotxx/2)/sqrtf(ns.x * ns.y * ns.z);
		}
		results[i].score = no;
	}
}

float FastHOGNMS::distqt(FastHOGPoint3 *p1, FastHOGPoint3 *p2)
{
	FastHOGPoint3 ns, b;
	ns.x = nsigma[0] * expf(p2->z); ns.y = nsigma[1] * expf(p2->z); ns.z = nsigma[2];
	b.x = p2->x - p1->x; b.y = p2->y - p1->y; b.z = p2->z - p1->z;
	b.x /= ns.x; b.y /= ns.y; b.z /= ns.z;
	return b.x * b.x + b.y * b.y + b.z * b.z;
}

void FastHOGNMS::shiftToMode(FastHOGPoint3* ms, FastHOGPoint3* at, float* wt, FastHOGPoint3 *tomode, int length)
{
	int i, count;
	FastHOGPoint3 ii,II;
	for (i=0; i<length; i++)
	{
		II = ms[i];;
		count = 0;

		do
		{
			ii = II;
			nvalue(&ii, &II, at, wt, length);
			++count;
		} while ( count < maxIterations && distqt(&ii,&II) > modeEpsilon );

		tomode[i].x = II.x; tomode[i].y = II.y; tomode[i].z = II.z;
	}
}

FastHOGResult* FastHOGNMS::ComputeNMSResults(FastHOGResult* formattedResults, int formattedResultsCount, bool *nmsResultsAvailable, int *nmsResultsCount,
									 int hWindowSizeX, int hWindowSizeY)
{
	if (!isAllocated)
	{
		wt = new float[hWindowSizeX * hWindowSizeX];
		at = new FastHOGPoint3[hWindowSizeX * hWindowSizeX];
		ms = new FastHOGPoint3[hWindowSizeX * hWindowSizeX];
		tomode = new FastHOGPoint3[hWindowSizeX * hWindowSizeX];
		nmsToMode = new FastHOGPoint3[hWindowSizeX * hWindowSizeX];
		nmsResults = new FastHOGResult[hWindowSizeX * hWindowSizeX];
		isAllocated = true;
	}

	int i, j;
	float cenx, ceny, nmsOK;

	*nmsResultsCount = 0;
	nmsResultsAvailable = false;

	for (i=0; i<formattedResultsCount; i++)
	{
		wt[i] = this->sigmoid(formattedResults[i].score);
		cenx = formattedResults[i].x + formattedResults[i].width / 2.0f;
		ceny = formattedResults[i].y + formattedResults[i].height / 2.0f;
		at[i] = FastHOGPoint3(cenx, ceny, logf(formattedResults[i].scale));
	}

	nvalue(ms, at, wt, formattedResultsCount);
	shiftToMode(ms, at, wt, tomode, formattedResultsCount);

	for (i=0; i<formattedResultsCount; i++)
	{
		nmsOK = true;
		for (j=0; j<*nmsResultsCount; j++)
		{
			if (distqt(&nmsToMode[j], &tomode[i]) < epsFinalDist)
			{
				nmsOK = false;
				break;
			}
		}

		if (nmsOK)
		{
			nmsResults[*nmsResultsCount].scale = expf(tomode[i].z);

			nmsResults[*nmsResultsCount].width = (int)floorf((float)hWindowSizeX * nmsResults[*nmsResultsCount].scale);
			nmsResults[*nmsResultsCount].height = (int)floorf((float)hWindowSizeY * nmsResults[*nmsResultsCount].scale);

			nmsResults[*nmsResultsCount].x = (int)ceilf(tomode[i].x - (float) hWindowSizeX * nmsResults[*nmsResultsCount].scale / 2);
			nmsResults[*nmsResultsCount].y = (int)ceilf(tomode[i].y - (float) hWindowSizeY * nmsResults[*nmsResultsCount].scale / 2);
	
			nmsToMode[*nmsResultsCount] = tomode[i];

			(*nmsResultsCount)++;
		}
	}

	fvalue(nmsToMode, nmsResults, *nmsResultsCount, at, wt, formattedResultsCount);

	return nmsResults;
}
