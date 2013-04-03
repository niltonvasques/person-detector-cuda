#ifndef __HOG_RESUL__
#define __HOG_RESUL__

namespace FastHOG_
{
	class FastHOGResult
	{
	public:
		float score;
		float scale;

		int width, height;
		int origX, origY;
		int x, y;

		FastHOGResult()
		{
			width = 0;
			height = 0;
			origX = 0;
			origY = 0;
			x = 0;
			y = 0;
		}
	};
}

#endif

