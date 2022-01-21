cmake_minimum_required (VERSION 3.10)
# Require version >= 3.15 to avoid setting the /W3 warning level in Visual Studio
# (see, e.g., https://gitlab.kitware.com/cmake/cmake/-/merge_requests/3250)

project(Assignment0 VERSION 1.315 LANGUAGES CXX)

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin )

# Make cache variables for install destinations
include(GNUInstallDirs)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)


# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)



# Turn on the ability to create folders to organize projects (.vcproj)
# It creates "CMakePredefinedTargets" folder by default and adds CMake
# defined projects like INSTALL.vcproj and ZERO_CHECK.vcproj
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Sub-directories where more CMakeLists.txt exist
add_subdirectory(cpp_src/main)
add_subdirectory(cpp_src/solver)

# Print out useful information
message("CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
message("CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message("CMAKE_CXX_CPPCHECK: " ${CMAKE_CXX_CPPCHECK})
message("CMAKE_CXX_CLANG_TIDY: " ${CMAKE_CXX_CLANG_TIDY})
message("CMAKE_CXX_INCLUDE_WHAT_YOU_USE: " ${CMAKE_CXX_INCLUDE_WHAT_YOU_USE})
message("CMAKE_CXX_COMPILER_LAUNCHER: " ${CMAKE_CXX_COMPILER_LAUNCHER})
message("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
message("EIGEN3_CXX_FLAGS: " ${EIGEN3_CXX_FLAGS})
message("CMAKE_INSTALL_INCLUDEDIR:" ${CMAKE_INSTALL_INCLUDEDIR})
message("CMAKE_INSTALL_LIBDIR:" ${CMAKE_INSTALL_LIBDIR})
message("CMAKE_INSTALL_BINDIR:" ${CMAKE_INSTALL_BINDIR})
message("CMAKE_SOURCE_DIR:" ${CMAKE_SOURCE_DIR})
