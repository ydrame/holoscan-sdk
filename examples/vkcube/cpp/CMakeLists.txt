# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(PROJECT_NAME vkcube)
include(GNUInstallDirs)
include(GenHeaderFromBinaryFile)
find_package(Vulkan REQUIRED)
find_package(X11 REQUIRED)

# compile GLSL source files to SPIR-V
include("${nvpro_core_CMAKE_DIR}/utilities.cmake")

set(GLSL_SOURCE_FILES)
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vkcube.glsl.vert")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vkcube.glsl.frag")


set(GLSLANGVALIDATOR ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})
compile_glsl(
    SOURCE_FILES ${GLSL_SOURCE_FILES}
    DST "${CMAKE_CURRENT_BINARY_DIR}/vulkan/spv"
    VULKAN_TARGET "vulkan1.2"
    HEADER ON
    )

set_source_files_properties(${GLSL_SOURCE_FILES} PROPERTIES GENERATED TRUE)

# Create examples
add_executable(vkcube
  vkcube.c
  cube.c
  esTransform.c
  main.cpp
  ${GLSL_SOURCE_FILES}
)

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
      ENABLE_XCB=1
    )

target_include_directories(${PROJECT_NAME}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(vkcube
  PRIVATE
  holoscan::core
  nvpro_core
  Vulkan::Vulkan
  X11::xcb
)

# Install examples

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(vkcube PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(FILES vkcube.cpp
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
endif()

# Install the compiled example
install(TARGETS vkcube
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
