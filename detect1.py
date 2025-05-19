# CMakeList.txt : CMake project for SuperResolution, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("SuperResolution" LANGUAGES CXX CUDA)


set(CMAKE_CUDA_COMPILER "D:\\software\\CUDA\\CUDA12.1\\bin\\nvcc.exe")
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75)
enable_language(CUDA)


###
#set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CXX_STANDARD 17)
# opencv
#set(OpenCV_DIR "D:/software/OpenCV/opencv4100/build")
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("D:/software/OpenCV/opencv4100/build/include"
                    "D:/software/OpenCV/opencv4100/build/include/opencv2")
link_directories("D:/software/OpenCV/opencv4100/build/x64/vc16/lib")
# libtorch
set(Torch_DIR "D:\\software\\LibTorch\\libtorch-win-shared-with-deps-debug-2.5.0+cu121\\share\\cmake\\Torch")
find_package(Torch REQUIRED)
#include_directories("D:/software/libtorch/include"
#                    "D:/software/libtorch/include/torch/csrc/api/include")
#link_directories("D:/software/libtorch/lib")
# CUDA
include_directories("D:\\software\\CUDA\\CUDA12.1\\include")
link_directories("D:\\software\\CUDA\\CUDA12.1\\lib\\x64")
# TensorRT
#include_directories("D:\\software\\TensorRT\\TensorRT-10.2.0.19.Windows.win10.cuda-12.5\\include")
#link_directories("D:\\software\\TensorRT\\TensorRT-10.2.0.19.Windows.win10.cuda-12.5\\lib")
#include_directories("D:\\software\\TensorRT\\TensorRT-10.3.0.26.Windows.win10.cuda-12.5\\include")
#link_directories("D:\\software\\TensorRT\\TensorRT-10.3.0.26.Windows.win10.cuda-12.5\\lib")
#include_directories("D:\\software\\TensorRT\\TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0\\include")
#link_directories("D:\\software\\TensorRT\\TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0\\lib")


# Add source to this project's executable.
add_executable (SuperResolution "SuperResolution.cpp" "SuperResolution.h" "main.cu")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET SuperResolution PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
# opencv
#target_link_libraries(Generation ${OpenCV_LIBS})
target_link_libraries(SuperResolution opencv_world4100d)
# libtorch
target_link_libraries(SuperResolution ${TORCH_LIBRARIES})
#target_link_libraries(SuperResolution torch torch_cuda torch_cpu c10_cuda c10 fbgemm asmjit)
# CUDA
target_link_libraries(SuperResolution cudart)
# TensorRT
#target_link_libraries(SuperResolution nvinfer nvinfer_plugin nvonnxparser)
#target_link_libraries(SuperResolution nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)