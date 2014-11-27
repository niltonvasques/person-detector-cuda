/*
 * =====================================================================================
 *
 *       Filename:  FileUtils.h
 *
 *    Description:  Provides file utility functions
 *
 *
 *         Author:  Nilton Vasques
 *        Company:  iVision UFBA
 *		  Created on: Jun 18, 2012
 *
 * =====================================================================================
 */

#ifndef FILE_UTIL_H
#define FILE_UTIL_H

#include <iostream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>

using namespace std;

void listFiles( string dirName , vector<string> &files )
{
#ifdef _WIN32
   WIN32_FIND_DATA FindFileData;
   HANDLE hFind;

   _tprintf (TEXT("Target file is %s\n"), dirName.c_str() );
   hFind = FindFirstFile(dirName.c_str(), &FindFileData);
   char lastFile[1000] = "";
   while( hFind != INVALID_HANDLE_VALUE && strcmp( lastFile, FindFileData.cFileName ) != 0){
	   strcpy( lastFile, FindFileData.cFileName );
	   files.push_back( string( lastFile ) );
		//_tprintf (TEXT("The first file found is %s\n"), FindFileData.cFileName);
		FindNextFile( hFind, &FindFileData );
		//getchar();
	}
	FindClose(hFind);
	sort( files.begin(), files.end() );
#else
	cout << "listFiles( dir, files ) Only Suported in Windows " << endl;
#endif
}



#endif 
