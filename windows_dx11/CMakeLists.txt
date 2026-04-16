cmake_minimum_required(VERSION 3.24)

project(mandelbrot_imgui LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS OFF)

include(FetchContent)

FetchContent_Declare(
  imgui
  URL https://github.com/ocornut/imgui/archive/refs/tags/v1.90.9.zip
)
FetchContent_MakeAvailable(imgui)

add_executable(compile_hlsl tools/compile_hlsl.cpp)
target_compile_definitions(compile_hlsl PRIVATE
  WIN32_LEAN_AND_MEAN
  NOMINMAX
)
target_compile_options(compile_hlsl PRIVATE
  -Wall
  -Wextra
)
target_link_libraries(compile_hlsl PRIVATE d3dcompiler)

set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(COMPUTE_SHADER_HEADER "${GENERATED_DIR}/mandelbrot_compute_shader.h")
add_custom_command(
  OUTPUT "${COMPUTE_SHADER_HEADER}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_DIR}"
  COMMAND "$<TARGET_FILE:compile_hlsl>"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/mandelbrot_compute.hlsl"
          "${COMPUTE_SHADER_HEADER}"
  DEPENDS compile_hlsl "${CMAKE_CURRENT_SOURCE_DIR}/src/mandelbrot_compute.hlsl"
  COMMENT "Compiling embedded Mandelbrot compute shader"
  VERBATIM
)

add_executable(mandelbrot_imgui WIN32
  src/main.cpp
  "${COMPUTE_SHADER_HEADER}"
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
)

target_include_directories(mandelbrot_imgui PRIVATE
  "${GENERATED_DIR}"
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)

target_compile_definitions(mandelbrot_imgui PRIVATE
  WIN32_LEAN_AND_MEAN
  NOMINMAX
)

target_compile_options(mandelbrot_imgui PRIVATE
  -Wall
  -Wextra
  -finput-charset=UTF-8
  -fexec-charset=UTF-8
)

target_link_options(mandelbrot_imgui PRIVATE
  -static
  -static-libgcc
  -static-libstdc++
)

target_link_libraries(mandelbrot_imgui PRIVATE
  quadmath
  winpthread
  d3d11
  dxgi
  d3dcompiler
  user32
  gdi32
  dwmapi
  imm32
)

add_custom_command(TARGET mandelbrot_imgui POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:mandelbrot_imgui>/portable"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:mandelbrot_imgui>" "$<TARGET_FILE_DIR:mandelbrot_imgui>/portable/mandelbrot_imgui.exe"
  COMMAND ${CMAKE_COMMAND} -E rm -f "$<TARGET_FILE_DIR:mandelbrot_imgui>/portable/D3DCompiler_47.dll"
  COMMENT "Creating single-exe portable release folder"
)
