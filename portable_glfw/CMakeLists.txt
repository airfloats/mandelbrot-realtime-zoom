cmake_minimum_required(VERSION 3.24)

if(APPLE)
  project(mandelbrot_glfw LANGUAGES CXX OBJCXX)
else()
  project(mandelbrot_glfw LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS OFF)

include(FetchContent)

FetchContent_Declare(
  glfw
  URL https://github.com/glfw/glfw/archive/refs/tags/3.4.zip
)
FetchContent_Declare(
  imgui
  URL https://github.com/ocornut/imgui/archive/refs/tags/v1.90.9.zip
)

if(UNIX AND NOT APPLE)
  set(GLFW_BUILD_X11 ON CACHE BOOL "Build GLFW with X11 support" FORCE)
  set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "Build GLFW with Wayland support" FORCE)
endif()

FetchContent_MakeAvailable(glfw imgui)

find_package(OpenGL REQUIRED)

set(MANDELBROT_APP_KIND)
set(MANDELBROT_PLATFORM_SOURCES)
if(APPLE)
  set(MANDELBROT_APP_KIND MACOSX_BUNDLE)
  list(APPEND MANDELBROT_PLATFORM_SOURCES src/macos_gestures.mm)
endif()

add_executable(mandelbrot_glfw ${MANDELBROT_APP_KIND}
  src/main.cpp
  ${MANDELBROT_PLATFORM_SOURCES}
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(mandelbrot_glfw PRIVATE
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)

target_compile_definitions(mandelbrot_glfw PRIVATE
  GL_SILENCE_DEPRECATION
)

target_compile_options(mandelbrot_glfw PRIVATE
  -Wall
  -Wextra
)

target_link_libraries(mandelbrot_glfw PRIVATE
  glfw
  OpenGL::GL
)

if(APPLE)
  set_target_properties(mandelbrot_glfw PROPERTIES
    MACOSX_BUNDLE_BUNDLE_NAME "Mandelbrot Realtime Zoom"
    MACOSX_BUNDLE_GUI_IDENTIFIER "io.github.airfloats.mandelbrot.realtimezoom"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    MACOSX_BUNDLE_BUNDLE_VERSION "1"
  )
endif()
