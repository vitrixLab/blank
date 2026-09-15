cmake_minimum_required(VERSION 3.14)
project(tiny_gpt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build with full optimisation
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(gpt gpt.cpp)

# Enable fast maths + vectorisation
if(MSVC)
  target_compile_options(gpt PRIVATE /O2 /fp:fast /W3)
else()
  target_compile_options(gpt PRIVATE -O2 -ffast-math -march=native -Wall -Wextra)
endif()

install(TARGETS gpt DESTINATION bin)
