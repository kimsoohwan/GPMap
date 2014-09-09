#ifndef _GPMAP_FILE_SYSTEM_HPP_
#define _GPMAP_FILE_SYSTEM_HPP_

// refer to http://www.boost.org/doc/libs/1_31_0/libs/filesystem/doc/index.htm

// STL
#include <string>
#include <vector>

// Boost
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
//#include <boost/regex.hpp>

namespace GPMap {

inline std::string extractFileExtension(const std::string &strFilePath)
{
	// boost path
	boost::filesystem::path p(strFilePath);

	// extension
	return p.extension().string();
}

inline std::string extractFileName(const std::string &strFilePath)
{
	// boost path
	boost::filesystem::path p(strFilePath);

	// extension
	return p.filename().string();
}


/** @brief Create a directory */
bool create_directory(const std::string strDirPath)
{
	return boost::filesystem::create_directory(strDirPath);
}

/** @brief Search a file from a directory and its subdirectories */
bool find_file_in_subdirectories(const std::string		&strDirPath,		// in this directory,
											const std::string		&strFileName,	// search for this name,
											std::string				&strFilePathFound)	// placing path here if found
{
	// current path
	boost::filesystem::path dir_path(strDirPath);

	// check if there exists the path
	if(!boost::filesystem::exists(dir_path)) return false;

	// for all items in the directory - files and subdirectories
	boost::filesystem::directory_iterator end_iter; // default construction yields past-the-end
	for(boost::filesystem::directory_iterator iter(dir_path); iter != end_iter; ++iter)
	{
		// if it is a sub-directory
		if(boost::filesystem::is_directory(*iter))
		{
			// search the file recursively
			if(find_file_in_subdirectories((*iter).path().string(), strFileName, strFilePathFound)) return true;
		}

		// if it is a file
		else if((*iter).path().filename() == strFileName) // see below
		{
			strFilePathFound = (*iter).path().string();
			return true;
		}
	}
	
	return false;
}

bool search_files(const std::string				&strDirPath,			// in this directory,
						const std::string				&strExtName,			// search for this extension,
						std::vector<std::string>	&strFileNameList)	// placing file names here if found
{
	// clear
	strFileNameList.clear();

	// filter
	//const boost::regex filter("somefiles.*\.txt");

	// current path
	boost::filesystem::path dir_path(strDirPath);

	// for all items in the directory - files and subdirectories
	boost::filesystem::directory_iterator end_iter; // default construction yields past-the-end
	for(boost::filesystem::directory_iterator iter(dir_path); iter != end_iter; ++iter)
	{
		 // skip if not a file
		 if(!boost::filesystem::is_regular_file(iter->status())) continue;
		 
		 // skip if no match
		 //boost::smatch what;
		 //if(!boost::regex_match((*iter).path().filename(), what, filter)) continue;
		 
		 // if it is a file
		 if((*iter).path().extension().string() == strExtName) // see below
		 
		 // file matches, store it
		 strFileNameList.push_back((*iter).path().filename().string());
	}

	return strFileNameList.size() > 0;
}

}

#endif