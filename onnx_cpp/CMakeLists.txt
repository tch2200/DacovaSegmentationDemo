cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Examples VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

add_executable(main main.cpp)
target_include_directories(main PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PUBLIC ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})


# cmake_minimum_required(VERSION 3.0)
# project(onnxruntime_cpp_inference_demo CXX C)

# set(onnxruntime_INSTALL_PREFIX /usr/local)
# set(onnxruntime_INCLUDE_DIRS
#   ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime
#   ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime/core/session
# )

# find_library(onnxruntime_LIBS NAMES onnxruntime PATHS /usr/local/lib)

# find_package(OpenCV REQUIRED)

# add_executable(${PROJECT_NAME}_app
#   ${PROJECT_SOURCE_DIR}/main.cpp
# #   ${PROJECT_SOURCE_DIR}/src/ort_session_handler.cpp
# )

# target_link_libraries(${PROJECT_NAME}_app
#   ${OpenCV_LIBRARIES}
#   ${onnxruntime_LIBS}
# )

# target_include_directories(${PROJECT_NAME}_app
#   SYSTEM PUBLIC
#     ${OpenCV_INCLUDE_DIRS}
#     ${onnxruntime_INCLUDE_DIRS}
# )