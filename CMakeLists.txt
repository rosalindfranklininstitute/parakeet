##############################################################################
# Build file for parakeet project
##############################################################################

cmake_minimum_required(VERSION 3.17.0)

# Set the project name
project(parakeet CXX)

# Set the build type to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set the cmake module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH. Required for submission to
# clusters which may not allow export of LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH True)

# Add pybind sub directory
add_subdirectory(pybind11)

# Add a C/C++ extension
pybind11_add_module(parakeet_ext 
  src/parakeet/freeze/freeze_ext.cc)

# Ensure we are using C++11
target_compile_features(parakeet_ext PUBLIC cxx_std_11)

# Set the include directory
target_include_directories(parakeet_ext PUBLIC src)

# Install the python extension
install(TARGETS parakeet_ext LIBRARY DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
