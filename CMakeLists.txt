cmake_minimum_required(VERSION 2.8)
project( Bird )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Bird testOpenCV.cpp )
target_link_libraries( Bird ${OpenCV_LIBS} )
