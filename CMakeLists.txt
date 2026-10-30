cmake_minimum_required(VERSION 3.16)

project(buff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(NNDEPLOYMENT_WITH_OPENVINO "Enable OpenVINO inference backend" ON)
option(NNDEPLOYMENT_WITH_TENSORRT "Enable TensorRT inference backend" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui dnn)
find_package(Threads REQUIRED)

if(NNDEPLOYMENT_WITH_OPENVINO)
    # OpenVINO 安装目录可用 -DOpenVINO_DIR=... 覆盖；未指定时按常见安装路径自动探测。
    if(NOT OpenVINO_DIR)
        file(GLOB _openvino_cmake_dirs "/opt/intel/openvino_*/runtime/cmake")
        if(_openvino_cmake_dirs)
            list(GET _openvino_cmake_dirs 0 OpenVINO_DIR)
        endif()
    endif()
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()

if(NNDEPLOYMENT_WITH_TENSORRT)
    message(WARNING "TensorRT 后端尚未实现，NNDEPLOYMENT_WITH_TENSORRT 暂不生效")
endif()

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(TASKS_DIR ${PROJECT_ROOT}/tasks)
set(INCLUDE_DIRS ${PROJECT_ROOT} ${TASKS_DIR})
set(OUTPUT_DIR ${PROJECT_ROOT}/result)

set(SOURCE_FILES
    ${PROJECT_ROOT}/main.cpp
    ${TASKS_DIR}/NNDetector.cpp
    ${TASKS_DIR}/interface.cpp
    ${TASKS_DIR}/deployment_lib.cpp
    ${TASKS_DIR}/postprocess.cpp
    ${TASKS_DIR}/performance.cpp
)

if(NNDEPLOYMENT_WITH_OPENVINO)
    list(APPEND SOURCE_FILES ${TASKS_DIR}/openvino.cpp)
endif()

add_executable(buff ${SOURCE_FILES})

target_include_directories(buff PRIVATE ${INCLUDE_DIRS})

target_link_libraries(buff PRIVATE
    opencv_core
    opencv_imgproc
    opencv_videoio
    opencv_highgui
    opencv_dnn
    Threads::Threads
)

if(NNDEPLOYMENT_WITH_OPENVINO)
    target_link_libraries(buff PRIVATE openvino::runtime)
endif()

target_compile_definitions(buff PRIVATE
    NNDEPLOYMENT_WITH_OPENVINO=$<BOOL:${NNDEPLOYMENT_WITH_OPENVINO}>
    NNDEPLOYMENT_WITH_TENSORRT=$<BOOL:${NNDEPLOYMENT_WITH_TENSORRT}>
    PROJECT_ROOT_DIR="${PROJECT_ROOT}"
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(buff PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

file(MAKE_DIRECTORY ${OUTPUT_DIR})

add_custom_command(TARGET buff POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:buff>/configs
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${PROJECT_ROOT}/configs $<TARGET_FILE_DIR:buff>/configs
    COMMENT "Copying configs to output directory"
)
