cmake_minimum_required(VERSION 3.5)
project(rest_server CXX)

include(FetchContent)
FetchContent_Declare(
        libcoro
        GIT_REPOSITORY https://github.com/jbaldwin/libcoro.git
        GIT_TAG        v0.3
)
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        8.1.1
)
FetchContent_MakeAvailable(libcoro)
FetchContent_MakeAvailable(fmt)

include(CheckIncludeFileCXX)

check_include_file_cxx(any HAS_ANY)
check_include_file_cxx(string_view HAS_STRING_VIEW)
check_include_file_cxx(coroutine HAS_COROUTINE)
check_include_file_cxx(format HAS_FORMAT)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_PREFIX_PATH "/media/sdd_1tb/miniconda3")

add_executable(${PROJECT_NAME} main.cc controllers/)

find_package(Drogon CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED HINTS "/media/sdd_1tb/miniconda3/lib/python3.9/site-packages/torch/share/cmake/")
find_package(TorchVision REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon fmt::fmt ${TORCH_LIBRARIES} "${OpenCV_LIBS}" TorchVision::TorchVision libcoro)
target_compile_options(${PROJECT_NAME} PRIVATE -std=c++20 -fcoroutines)

message(STATUS "use c++20")

aux_source_directory(controllers CTL_SRC)
aux_source_directory(filters FILTER_SRC)
aux_source_directory(plugins PLUGIN_SRC)
aux_source_directory(models MODEL_SRC)
aux_source_directory(src LIB_SRC)

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
