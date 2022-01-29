cmake_minimum_required(VERSION 3.5)
project(rest_server CXX)

include(cmake/CPM.cmake)
CPMAddPackage(
        NAME libcoro
        VERSION 0.3
        GITHUB_REPOSITORY "jbaldwin/libcoro"
        GIT_TAG "v0.3"
        OPTIONS
            "LIBCORO_BUILD_TESTS OFF"
            "LIBCORO_BUILD_EXAMPLES OFF"
)
CPMAddPackage(
        NAME fmt
        VERSION 8.1.1
        GITHUB_REPOSITORY "fmtlib/fmt"
        GIT_TAG "8.1.1"
)

include(CheckIncludeFileCXX)

check_include_file_cxx(any HAS_ANY)
check_include_file_cxx(string_view HAS_STRING_VIEW)
check_include_file_cxx(coroutine HAS_COROUTINE)
check_include_file_cxx(format HAS_FORMAT)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

add_executable(${PROJECT_NAME} main.cc controllers/)

find_package(Drogon CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon fmt::fmt ${TORCH_LIBRARIES} "${OpenCV_LIBS}" TorchVision::TorchVision libcoro)
target_compile_options(${PROJECT_NAME} PRIVATE -std=c++20 -fcoroutines)

message(STATUS "use c++20")

aux_source_directory(controllers CTL_SRC)
aux_source_directory(filters FILTER_SRC)
aux_source_directory(plugins PLUGIN_SRC)
aux_source_directory(models MODEL_SRC)
aux_source_directory(lib LIB_SRC)

drogon_create_views(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/views
                    ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(${PROJECT_NAME}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                   ${CMAKE_CURRENT_SOURCE_DIR}/models)
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CTL_SRC}
               ${FILTER_SRC}
               ${PLUGIN_SRC}
               ${MODEL_SRC}
                ${LIB_SRC})

add_subdirectory(test)
