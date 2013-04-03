#ifndef __HOG_VECTOR_3D__
#define __HOG_VECTOR_3D__

namespace FastHOG_
{
	class FastHOGPoint3
	{
	public:
		float x,y,z;

		FastHOGPoint3(float x, float y, float z) { this->x = x; this->y = y; this->z = z; }
		FastHOGPoint3() { this->x = 0; this->y = 0; this->z = 0; }
	};
}

#endif

