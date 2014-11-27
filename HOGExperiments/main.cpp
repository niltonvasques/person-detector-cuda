/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  Project HOGExperiments, created for studies HOG and parallel this algorithm
 *					
 *
 *
 *         Author:  Nilton Vasques
 *        Company:  iVision UFBA
 *		  Created on: Jun 18, 2012
 *
 * =====================================================================================
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <boost/timer.hpp>

#include "DatasetPaths.h"
#include "FileUtils.h"
#include "NonMaxSupression.h"
#include "Slide.h"
#include "SlidingOptions.h"
#include "Scale.h"
#include "Window.h"
#include "SimplePersonDetector.h"
#include "HOGCuda.h"
#include "SaliencyCUDA.h"

#ifdef _WIN32
	#include <Windows.h>
#endif

#include "HOGOpenCv.h"

using namespace std;
using namespace cv;

#define SCOPE
#define MULTISCALE

int main(){

	DetectionData slideall_data;

	vector< string > files;
	string path = DATASET_PETS_2009_VIEW_007;
	path.append( "*.jpg" );
	listFiles( path, files );
	assert( files.size() > 0 );

	//IplImage* ip = cvLoadImage( "D://metro.jpg" );
	//cv::Mat3b im = cv::Mat3b::zeros( ip->height, ip->width );
	//im.data = (unsigned char*) ip->imageData;

	//SaliencyCUDA saliency;
	//Cuda::Device::GpuImage<uchar> im_( Cuda::Size( ip->width, ip->height ), im.channels() );
	//im_.upload( im.data );
	//Cuda::Device::GpuImage<float> sal = saliency.Saliency( im_ );
	//
	//sal.download();
	//float* data = sal.host_ptr(0);

	//Mat1f mat = Mat1f( sal.size.height, sal.size.width );
	//mat.data = (uchar*)data;
	//imshow( "Saliency", mat);
	//imshow( "Image", im);

	//cout << "Saliency Time: " << saliency.elapsed_time << endl;
	//waitKey();

	//sal.release();
	//im.release();


//
//	//Detector &detector = CudaPersonDetectorByPart();
//	//Detector &detector = CudaPersonDetectorByPart( im.cols, im.rows );
//	//Detector &detector2 = CudaPersonDetectorByPart();
//	//Detector &detector2 = OpenCVPersonDetector();
//	//Detector &detector = CudaPersonDetectorOneThreadBlock();
	Detector &detector = FastHOGPersonDetector();
//	//Detector &detector = SimplePersonDetector();
//
	std::ofstream infostream( "D:\\Performance\\planilha.csv" );

	infostream << " Método "<< " ; " << " Numéro de Janelas ;"<<" Número de Escalas; " <<" Resolução da Imagem; " <<" Tempo Total " << " \t" << endl;


	/**
		 * Normal sliding window
		 */
	std::vector<cv::Rect> slideall_misses;
	SCOPE
	{
		SlidingOptions opts;
		opts.scale_factor = 0.95f;
		for( int i = 0; i < files.size(); i++ ){
			path = DATASET_PETS_2009_VIEW_007;
			path.append( files.at(i) );

			IplImage* ip = cvLoadImage( path.c_str());
			cv::Mat3b im = cv::Mat3b::zeros( ip->height, ip->width );
			im.data = (unsigned char*) ip->imageData;

			double total_time = 0.0;
			double total_time2 = 0.0;
			std::vector<cv::Rect> slideall_detections;
			std::vector<cv::Rect> slideall_detections2;

			boost::timer slide_timer;

			int total_windows = detector.PredictMultiscale( im, slideall_detections, opts );
			//detector.PredictMultiWindow( im, slideall_detections );

			total_time += slide_timer.elapsed();

			//detector.options( opts );

			infostream << detector.getDetectorName() << ";"<< opts.number_windows << ";" << opts.scales_count << ";" << im.cols << "X" << im.rows << ";" << total_time*1000 << "\t" << endl;

			//boost::timer slide_timer2;

			//detector2.PredictMultiscale( im, slideall_detections2 );
			////detector2.PredictMultiWindow( im, slideall_detections2 );

			//total_time2 += slide_timer2.elapsed();

			//	nwindows   += NumberOfWindows(imresized,opts);

			////NonMaxSupressionStats slideall_nms_stats = Perfomance(slideall_nms_results, rects[index], im);

			//This Draw Rects in all scales founded.
			for( vector<Rect>::iterator it = slideall_detections.begin(); it != slideall_detections.end(); it++ ){
					rectangle( im, *it, Scalar( 0, 255, 0 ) );
			}

			//for( vector<Rect>::iterator it = slideall_detections2.begin(); it != slideall_detections2.end(); it++ ){
			//		rectangle( im, *it, Scalar( 0, 0, 255 ) );
			//}


		
			//cout << "Tempo total GPU: " << total_time << endl;
			//cout << "Tempo total CPU: " << total_time2 << endl;
			//cout << "Numero de Windows: " << nwindows << endl;
			imshow( "Result", im );
			waitKey(1);

			//opts.scale_factor -= 0.05f;
			if( opts.scale_factor < 0 ) break;
		}

		//grid.Put(slideall_nms_stats.display, 0, 0);


		//slideall_misses.insert(
		//		slideall_misses.end(),
		//		slideall_nms_stats.misses.begin(),
		//		slideall_nms_stats.misses.end()
		//	);

		/*infostream << "\"" << "SLIDEALL" << "\"" << "\t"
					<< "\"" << *it << "\"" << "\t"
					<< "\"" << slideall_detections.size() << "\"" << "\t"
					<< "\"" << slideall_nms_stats.falseAlarm.rects.size() << "\"" << "\t"
					<< "\"" << slideall_nms_stats.match.rects.size() << "\"" << "\t"
					<< "\"" << slideall_nms_stats.misses.size() << "\"" << "\t"
					<< "\"" << total_time << "\"" << "\t"
					<< "\"" << total_time / nwindows << "\"" << "\t"
					<< "\"" << nwindows << "\"" << "\t"
					<< "\"" << 0 << "\"" << "\t"
					<< std::endl;*/

		/*slideall_data.tp.push_back(slideall_nms_stats.match.rects.size());
		slideall_data.tp_neighbors.insert(
				slideall_data.tp_neighbors.end(),
				slideall_nms_stats.match.neighbors.begin(),
				slideall_nms_stats.match.neighbors.end()
			);

		slideall_data.fp.push_back(slideall_nms_stats.falseAlarm.rects.size());
		slideall_data.fp_neighbors.insert(
				slideall_data.fp_neighbors.end(),
				slideall_nms_stats.falseAlarm.neighbors.begin(),
				slideall_nms_stats.falseAlarm.neighbors.end()
			);

		slideall_data.fn.push_back(slideall_nms_stats.misses.size());*/
	}
//#ifdef _WIN32
//	Beep(50,1000);
//#endif
//	waitKey();
//
//	Cuda::HOGCuda::destroyInstance();
}
