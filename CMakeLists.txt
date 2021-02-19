cmake_minimum_required(VERSION 3.11.0)

# Set the project name
project(amplus CXX)

# Set the cmake module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH. Required for submission to
# clusters which may not allow export of LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH True)

# Find Pybind11
find_package(pybind11 REQUIRED)

# Add a C/C++ extension
pybind11_add_module(freeze_ext 
  src/amplus/freeze/freeze_ext.cc)

# Ensure we are using C++11
target_compile_features(freeze_ext PUBLIC cxx_std_11)

# Set the include directory
target_include_directories(freeze_ext PUBLIC src)

# Set the coverage compile and link flags for gcc or clang
set(COVERAGE_COMPILE_FLAGS -coverage)
set(COVERAGE_LINK_FLAGS    -coverage)

# Set the coverage compile and link flags in the debug build
target_compile_options(freeze_ext 
  PUBLIC "$<$<CONFIG:DEBUG>:${COVERAGE_COMPILE_FLAGS}>")
target_link_options(freeze_ext 
  PUBLIC "$<$<CONFIG:DEBUG>:${COVERAGE_LINK_FLAGS}>")

# Install the python extension
install(TARGETS freeze_ext LIBRARY DESTINATION src/amplus/freeze)
