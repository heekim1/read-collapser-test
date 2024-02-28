include("../cmake/catch2.cmake")

set(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
set(GCC_COVERAGE_LINK_FLAGS "--coverage")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

add_executable(ClustererTests clusterer-tests.cpp majority-voting-consensus-generator-tests.cpp read-sampler-tests.cpp
        consensus/cnn-consensus-strategy-tests.cpp msa_bam/bam-to-msa-converter-tests.cpp gzip-fastq-sink-tests.cpp
        consensus/deep-learning-consensus-caller-tests.cpp)
target_link_libraries(ClustererTests PRIVATE armadillo onnxruntime Catch2::Catch2WithMain clusterer_lib spdlog::spdlog ${Boost_LIBRARIES} ${ARMADILLO_LIBRARIES} ${ONNX_LIBRARIES})
target_include_directories(ClustererTests PUBLIC ${HTSlib_INCLUDE_DIRS})

target_compile_definitions(ClustererTests PUBLIC TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../resources/")
