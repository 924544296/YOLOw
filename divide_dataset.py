# CMakeList.txt: DeepLearning 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("DeepLearning" LANGUAGES CXX CUDA)
# CUDA
set(CMAKE_CUDA_COMPILER "D:\\software\\CUDA\\CUDA12.1\\bin\\nvcc.exe")
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75)
enable_language(CUDA)
#set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CXX_STANDARD 17)
include_directories("D:\\software\\CUDA\\CUDA12.1\\include")
link_directories("D:\\software\\CUDA\\CUDA12.1\\lib\\x64")
# opencv
#set(OpenCV_DIR "D:/software/opencv/opencv490/build")
#find_package(OpenCV REQUIRED)
include_directories("D:/software/OpenCV/opencv4100/build/include"
                    "D:/software/OpenCV/opencv4100/build/include/opencv2")
link_directories("D:/software/OpenCV/opencv4100/build/x64/vc16/lib")
# libtorch
set(Torch_DIR "D:\\software\\LibTorch\\libtorch-win-shared-with-deps-debug-2.5.0+cu121\\share\\cmake\\Torch")
find_package(Torch REQUIRED)
#target_include_directories(DeepLearning ${Torch_INCLUDE_DIRS})
#include_directories("D:/software/LibTorch/libtorch-win-shared-with-deps-debug-2.5.0+cu121/include"
#                    "D:/software/LibTorch/libtorch-win-shared-with-deps-debug-2.5.0+cu121/include/torch/csrc/api/include")
#link_directories("D:/software/LibTorch/libtorch-win-shared-with-deps-debug-2.5.0+cu121/lib")


# 将源代码添加到此项目的可执行文件。
add_executable (DeepLearning "main.cu" "functions.h" "functions.cpp" "networkstructure.h" "networkstructure.cpp" "dataset_.h" "dataset_.cpp" "tinyxml2.h" "tinyxml2.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET DeepLearning PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
# CUDA
target_link_libraries(DeepLearning cudart)
# opencv
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(DeepLearning ${OpenCV_LIBS})
target_link_libraries(DeepLearning opencv_world4100d)
# libtorch
#target_include_directories(DeepLearning ${Torch_INCLUDE_DIRS})
target_link_libraries(DeepLearning ${TORCH_LIBRARIES})
#target_link_libraries(DeepLearning asmjit c10 c10_cuda cudnn64_9 fbgemm torch_cpu torch_cuda uv)









