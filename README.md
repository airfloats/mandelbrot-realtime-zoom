# Mandelbrot Realtime Zoom

实时曼德布罗集放大工具。目标是用当前屏幕分辨率实时计算当前视野，不保存历史大图；交互时先显示可用预览，停止移动后逐步补到清晰帧。

## Platforms

- Windows x64: `windows_dx11`，Win32 + DirectX 11 + Dear ImGui，使用 GCC `__float128` / quadmath 做高精度坐标和迭代。
- Linux x64: `portable_glfw`，GLFW + OpenGL + Dear ImGui，CPU 多线程渲染。
- Linux arm64: `portable_glfw`，GLFW + OpenGL + Dear ImGui，CPU 多线程渲染。
- macOS arm64 / x64: `portable_glfw`，GLFW + OpenGL + Dear ImGui，CPU 多线程渲染。

macOS 和 Linux 版本优先保证可运行。它们目前用 `long double`，深度放大精度低于 Windows `__float128` 版本。后续要做同等级深度放大，应接入 MPFR 或 Boost.Multiprecision。

macOS release 包使用 ad-hoc 签名，不是 Apple Developer ID 公证包。如果 macOS 仍提示“已损坏，无法打开”，在终端进入解压目录后执行：

```bash
xattr -dr com.apple.quarantine mandelbrot_glfw.app
open mandelbrot_glfw.app
```

## GitHub Builds

仓库带有 GitHub Actions：

- `windows-dx11-x64`: 生成 Windows 便携 exe。
- `linux-glfw-x64`: 生成 Linux 可执行文件压缩包。
- `linux-glfw-arm64`: 生成 Linux arm64 可执行文件压缩包。
- `macos-glfw-arm64`: 生成 Apple Silicon `.app`。
- `macos-glfw-x64`: 生成 Intel macOS `.app`。

每次 push、pull request 或手动运行 workflow 后，在 Actions 页面下载 artifacts。

## Local Windows Build

```powershell
cd windows_dx11
$env:PATH='C:\msys64\mingw64\bin;C:\msys64\usr\bin;' + $env:PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

输出：

```text
windows_dx11/build/portable/mandelbrot_imgui.exe
```

## Local GLFW Build

Linux/macOS:

```bash
cd portable_glfw
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

macOS arm64:

```bash
cmake -S portable_glfw -B build/macos-arm64 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build/macos-arm64 --config Release
open build/macos-arm64/mandelbrot_glfw.app
```

## Controls

- 左键点击：选择放大目标。
- 左键拖动：平移视图。
- 滚轮：手动缩放，先直接放大当前画面，后台再补清晰。
- 配色：包含高对比霓虹色带，适合观察边界细节。
- 空格：开始或暂停自动放大。
- R：重置视图。
- Esc：退出。
