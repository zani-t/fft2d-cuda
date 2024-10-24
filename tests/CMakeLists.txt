# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA language
enable_language(CUDA)

# Include directories
include_directories(${PROJECT_SOURCE_DIR})

# Add test executable
add_executable(runUnitTests test_fft.cu)

# Link libraries
target_link_libraries(runUnitTests fft_library gtest gtest_main pthread)

# Add test to CTest
add_test(NAME FFTUnitTests COMMAND runUnitTests)

# Set the working directory for the test
set_tests_properties(FFTUnitTests PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/tests")
