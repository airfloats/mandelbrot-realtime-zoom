#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

namespace {

using Clock = std::chrono::steady_clock;
using Real = long double;

constexpr int kMaxIterationLimit = 2000000;
constexpr int kTileWidth = 32;
constexpr int kTileHeight = 8;

struct RenderRequest {
  int width = 1;
  int height = 1;
  int maxIterations = 1000;
  int workerCount = 1;
  int palette = 0;
  bool smoothColor = true;
  uint64_t sequence = 0;
  uint64_t viewRevision = 0;
  Real centerX = -0.5L;
  Real centerY = 0.0L;
  Real zoom = 1.0L;
};

struct RenderResult {
  int width = 0;
  int height = 0;
  int maxIterations = 0;
  uint64_t sequence = 0;
  uint64_t viewRevision = 0;
  bool aborted = false;
  double milliseconds = 0.0;
  std::vector<uint32_t> pixels;
};

struct AppState {
  Real centerX = -0.5L;
  Real centerY = 0.0L;
  Real targetX = -0.5L;
  Real targetY = 0.0L;
  Real zoom = 1.0L;
  int maxIterations = 5000;
  int previewIterations = 700;
  int cpuThreads = 1;
  int palette = 0;
  float zoomSpeed = 1.75f;
  bool autoZoom = false;
  bool autoIterations = true;
  bool smoothColor = true;
  bool mouseDown = false;
  bool dragging = false;
  ImVec2 mouseDownPos = ImVec2(0.0f, 0.0f);
  Real dragCenterX = -0.5L;
  Real dragCenterY = 0.0L;
  double lastInteractionSeconds = 0.0;
  uint64_t viewRevision = 1;
  uint64_t displayedViewRevision = 0;
  uint64_t submittedViewRevision = 0;
  bool dirty = true;
};

struct FrameStats {
  int displayedWidth = 0;
  int displayedHeight = 0;
  int displayedIterations = 0;
  double renderMilliseconds = 0.0;
  uint64_t displayedSequence = 0;
};

uint8_t toByte(double value) {
  return static_cast<uint8_t>(std::clamp(value, 0.0, 255.0) + 0.5);
}

uint32_t packRgba(uint8_t r, uint8_t g, uint8_t b) {
  return static_cast<uint32_t>(r) |
         (static_cast<uint32_t>(g) << 8u) |
         (static_cast<uint32_t>(b) << 16u) |
         (0xffu << 24u);
}

uint32_t paletteColor(double t, int palette) {
  t = std::clamp(t, 0.0, 1.0);
  constexpr double pi2 = 6.2831853071795864769;
  if (palette == 1) {
    return packRgba(toByte(255.0 * std::pow(t, 0.32)), toByte(190.0 * t), toByte(80.0 + 120.0 * (1.0 - t)));
  }
  if (palette == 2) {
    const double band = 0.5 + 0.5 * std::cos(pi2 * t * 12.0);
    const double base = 255.0 * std::pow(t, 0.22);
    return packRgba(toByte(base), toByte(base * (0.72 + 0.28 * band)), toByte(base * 0.78));
  }
  const double r = 128.0 + 127.0 * std::cos(pi2 * (t + 0.04));
  const double g = 128.0 + 127.0 * std::cos(pi2 * (t + 0.34));
  const double b = 128.0 + 127.0 * std::cos(pi2 * (t + 0.67));
  return packRgba(toByte(r), toByte(g), toByte(b));
}

uint32_t mandelbrotPixel(Real cx, Real cy, int maxIterations, int palette, bool smoothColor) {
  const Real xMinusQuarter = cx - 0.25L;
  const Real q = xMinusQuarter * xMinusQuarter + cy * cy;
  if (q * (q + xMinusQuarter) <= 0.25L * cy * cy) {
    return packRgba(4, 5, 5);
  }
  const Real xPlusOne = cx + 1.0L;
  if (xPlusOne * xPlusOne + cy * cy <= 0.0625L) {
    return packRgba(4, 5, 5);
  }

  Real zx = 0.0L;
  Real zy = 0.0L;
  Real zx2 = 0.0L;
  Real zy2 = 0.0L;
  int iter = 0;
  for (; iter < maxIterations; ++iter) {
    if (zx2 + zy2 > 4.0L) {
      break;
    }
    zy = 2.0L * zx * zy + cy;
    zx = zx2 - zy2 + cx;
    zx2 = zx * zx;
    zy2 = zy * zy;
  }

  if (iter >= maxIterations) {
    return packRgba(4, 5, 5);
  }

  Real value = static_cast<Real>(iter);
  if (smoothColor && iter > 0) {
    const Real magnitude2 = std::max(zx2 + zy2, static_cast<Real>(4.0000000001L));
    const Real logZn = 0.5L * std::log(static_cast<double>(magnitude2));
    const Real nu = std::log(static_cast<double>(logZn / std::log(2.0))) / std::log(2.0);
    value = static_cast<Real>(iter) + 1.0L - nu;
  }
  const double t = std::sqrt(static_cast<double>(std::max(value, static_cast<Real>(0.0L)) / maxIterations));
  return paletteColor(t, palette);
}

int tileCount(const RenderRequest& request) {
  const int cols = (request.width + kTileWidth - 1) / kTileWidth;
  const int rows = (request.height + kTileHeight - 1) / kTileHeight;
  return std::max(1, cols * rows);
}

RenderResult renderMandelbrot(const RenderRequest& request, const std::atomic<uint64_t>& latestSequence, std::atomic<int>& progressDone) {
  RenderResult result;
  result.width = request.width;
  result.height = request.height;
  result.maxIterations = request.maxIterations;
  result.sequence = request.sequence;
  result.viewRevision = request.viewRevision;
  result.pixels.resize(static_cast<size_t>(request.width) * static_cast<size_t>(request.height));

  const auto start = Clock::now();
  const Real aspect = static_cast<Real>(request.width) / static_cast<Real>(std::max(1, request.height));
  const Real halfHeight = 1.5L / request.zoom;
  const Real halfWidth = halfHeight * aspect;
  const Real stepX = (2.0L * halfWidth) / static_cast<Real>(request.width);
  const Real stepY = (2.0L * halfHeight) / static_cast<Real>(request.height);
  const Real left = request.centerX - halfWidth + 0.5L * stepX;
  const Real top = request.centerY + halfHeight - 0.5L * stepY;

  const int workerCount = std::clamp(request.workerCount, 1, 64);
  const int cols = (request.width + kTileWidth - 1) / kTileWidth;
  const int rows = (request.height + kTileHeight - 1) / kTileHeight;
  const int totalTiles = std::max(1, cols * rows);
  std::atomic<int> nextTile = 0;
  std::atomic<bool> canceled = false;
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(workerCount));

  auto isCanceled = [&]() {
    return latestSequence.load(std::memory_order_relaxed) != request.sequence;
  };

  auto renderTiles = [&]() {
    for (;;) {
      if (isCanceled()) {
        canceled.store(true, std::memory_order_relaxed);
        return;
      }
      const int tile = nextTile.fetch_add(1, std::memory_order_relaxed);
      if (tile >= totalTiles) {
        return;
      }
      const int tileX = tile % cols;
      const int tileY = tile / cols;
      const int x0 = tileX * kTileWidth;
      const int y0 = tileY * kTileHeight;
      const int x1 = std::min(request.width, x0 + kTileWidth);
      const int y1 = std::min(request.height, y0 + kTileHeight);
      for (int y = y0; y < y1; ++y) {
        uint32_t* row = result.pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(request.width);
        const Real cy = top - static_cast<Real>(y) * stepY;
        for (int x = x0; x < x1; ++x) {
          if ((x & 31) == 0 && isCanceled()) {
            canceled.store(true, std::memory_order_relaxed);
            return;
          }
          const Real cx = left + static_cast<Real>(x) * stepX;
          row[x] = mandelbrotPixel(cx, cy, request.maxIterations, request.palette, request.smoothColor);
        }
      }
      progressDone.fetch_add(1, std::memory_order_relaxed);
    }
  };

  for (int i = 0; i < workerCount; ++i) {
    workers.emplace_back(renderTiles);
  }
  for (std::thread& worker : workers) {
    worker.join();
  }

  const auto end = Clock::now();
  result.aborted = canceled.load(std::memory_order_relaxed) || isCanceled();
  result.milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
  return result;
}

class CpuRenderer {
 public:
  CpuRenderer() : worker_([this]() { workerLoop(); }) {}
  ~CpuRenderer() {
    {
      std::lock_guard<std::mutex> lock(requestMutex_);
      stop_ = true;
    }
    requestCv_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void submit(RenderRequest request) {
    latestSequence_.store(request.sequence, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(requestMutex_);
      pendingRequest_ = request;
    }
    requestCv_.notify_one();
  }

  void cancel(uint64_t sequence) {
    latestSequence_.store(sequence, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(requestMutex_);
      pendingRequest_.reset();
    }
    requestCv_.notify_one();
  }

  bool takeResult(RenderResult& result) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    if (!completedResult_) {
      return false;
    }
    result = std::move(*completedResult_);
    completedResult_.reset();
    return true;
  }

  bool busy() const {
    return working_.load(std::memory_order_relaxed);
  }

  float progress() const {
    const int total = progressTotal_.load(std::memory_order_relaxed);
    if (total <= 0) {
      return working_.load(std::memory_order_relaxed) ? 0.0f : 1.0f;
    }
    const int done = progressDone_.load(std::memory_order_relaxed);
    return std::clamp(static_cast<float>(done) / static_cast<float>(total), 0.0f, 1.0f);
  }

 private:
  void workerLoop() {
    for (;;) {
      RenderRequest request;
      {
        std::unique_lock<std::mutex> lock(requestMutex_);
        requestCv_.wait(lock, [this]() { return stop_ || pendingRequest_.has_value(); });
        if (stop_) {
          break;
        }
        request = *pendingRequest_;
        pendingRequest_.reset();
      }

      working_.store(true, std::memory_order_relaxed);
      progressDone_.store(0, std::memory_order_relaxed);
      progressTotal_.store(tileCount(request), std::memory_order_relaxed);
      RenderResult result = renderMandelbrot(request, latestSequence_, progressDone_);
      working_.store(false, std::memory_order_relaxed);
      progressTotal_.store(0, std::memory_order_relaxed);
      if (result.aborted || result.sequence != latestSequence_.load(std::memory_order_relaxed)) {
        continue;
      }
      {
        std::lock_guard<std::mutex> lock(resultMutex_);
        completedResult_ = std::move(result);
      }
    }
  }

  std::thread worker_;
  std::mutex requestMutex_;
  std::condition_variable requestCv_;
  std::optional<RenderRequest> pendingRequest_;
  bool stop_ = false;
  std::atomic<bool> working_ = false;
  std::atomic<uint64_t> latestSequence_ = 0;
  std::atomic<int> progressDone_ = 0;
  std::atomic<int> progressTotal_ = 0;
  std::mutex resultMutex_;
  std::optional<RenderResult> completedResult_;
};

std::string formatReal(Real value, int precision = 12) {
  char buffer[96] = {};
  std::snprintf(buffer, sizeof(buffer), "%.*Le", precision, value);
  return buffer;
}

void resetView(AppState& state, double nowSeconds) {
  state.centerX = -0.5L;
  state.centerY = 0.0L;
  state.targetX = state.centerX;
  state.targetY = state.centerY;
  state.zoom = 1.0L;
  state.autoZoom = false;
  state.mouseDown = false;
  state.dragging = false;
  state.lastInteractionSeconds = nowSeconds;
  state.viewRevision += 1;
  state.dirty = true;
}

Real screenToWorldX(float mouseX, int width, int height, const AppState& state) {
  const Real aspect = static_cast<Real>(width) / static_cast<Real>(std::max(1, height));
  const Real halfWidth = (1.5L / state.zoom) * aspect;
  const Real nx = (static_cast<Real>(mouseX) / static_cast<Real>(std::max(1, width))) * 2.0L - 1.0L;
  return state.centerX + nx * halfWidth;
}

Real screenToWorldY(float mouseY, int height, const AppState& state) {
  const Real halfHeight = 1.5L / state.zoom;
  const Real ny = 1.0L - (static_cast<Real>(mouseY) / static_cast<Real>(std::max(1, height))) * 2.0L;
  return state.centerY + ny * halfHeight;
}

int chooseIterations(const AppState& state, bool moving) {
  int iterations = state.maxIterations;
  if (state.autoIterations) {
    const double logZoom = std::max(0.0, std::log10(static_cast<double>(std::max(state.zoom, static_cast<Real>(1.0L)))));
    iterations = std::clamp(360 + static_cast<int>(logZoom * 220.0), 320, state.maxIterations);
  }
  if (moving) {
    iterations = std::min(iterations, state.previewIterations);
  }
  return std::clamp(iterations, 32, kMaxIterationLimit);
}

void updateAutoZoom(AppState& state, double dt) {
  if (!state.autoZoom) {
    return;
  }
  const Real follow = 1.0L - std::exp(static_cast<Real>(-dt * 5.0));
  state.centerX += (state.targetX - state.centerX) * follow;
  state.centerY += (state.targetY - state.centerY) * follow;
  state.zoom *= std::pow(static_cast<Real>(state.zoomSpeed), static_cast<Real>(dt));
  state.viewRevision += 1;
  state.dirty = true;
}

void handleInput(AppState& state, int width, int height, double nowSeconds) {
  ImGuiIO& io = ImGui::GetIO();
  const bool mouseInWindow = io.MousePos.x >= 0.0f && io.MousePos.y >= 0.0f &&
                             io.MousePos.x < static_cast<float>(width) &&
                             io.MousePos.y < static_cast<float>(height);
  const bool canvasInput = mouseInWindow && !io.WantCaptureMouse;

  if (canvasInput && io.MouseWheel != 0.0f) {
    const Real beforeX = screenToWorldX(io.MousePos.x, width, height, state);
    const Real beforeY = screenToWorldY(io.MousePos.y, height, state);
    const Real factor = std::pow(static_cast<Real>(1.18L), static_cast<Real>(io.MouseWheel));
    state.zoom = std::max(static_cast<Real>(0.05L), state.zoom * factor);
    const Real afterX = screenToWorldX(io.MousePos.x, width, height, state);
    const Real afterY = screenToWorldY(io.MousePos.y, height, state);
    state.centerX += beforeX - afterX;
    state.centerY += beforeY - afterY;
    state.targetX = state.centerX;
    state.targetY = state.centerY;
    state.autoZoom = false;
    state.lastInteractionSeconds = nowSeconds;
    state.viewRevision += 1;
    state.dirty = true;
  }

  if (canvasInput && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    state.mouseDown = true;
    state.dragging = false;
    state.mouseDownPos = io.MousePos;
    state.dragCenterX = state.centerX;
    state.dragCenterY = state.centerY;
  }

  if (state.mouseDown && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    const float dx = io.MousePos.x - state.mouseDownPos.x;
    const float dy = io.MousePos.y - state.mouseDownPos.y;
    if (dx * dx + dy * dy > 16.0f) {
      state.dragging = true;
    }
    if (state.dragging) {
      const Real aspect = static_cast<Real>(width) / static_cast<Real>(std::max(1, height));
      state.centerX = state.dragCenterX - static_cast<Real>(dx) * (3.0L / state.zoom * aspect) / static_cast<Real>(std::max(1, width));
      state.centerY = state.dragCenterY + static_cast<Real>(dy) * (3.0L / state.zoom) / static_cast<Real>(std::max(1, height));
      state.targetX = state.centerX;
      state.targetY = state.centerY;
      state.autoZoom = false;
      state.lastInteractionSeconds = nowSeconds;
      state.viewRevision += 1;
      state.dirty = true;
    }
  }

  if (state.mouseDown && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    if (!state.dragging && canvasInput) {
      state.targetX = screenToWorldX(io.MousePos.x, width, height, state);
      state.targetY = screenToWorldY(io.MousePos.y, height, state);
      state.autoZoom = true;
      state.lastInteractionSeconds = nowSeconds;
      state.viewRevision += 1;
      state.dirty = true;
    }
    state.mouseDown = false;
    state.dragging = false;
  }
}

void drawControls(AppState& state, const FrameStats& stats, bool busy, float progress, double nowSeconds) {
  ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(390.0f, 0.0f), ImGuiCond_FirstUseEver);
  ImGui::Begin("Mandelbrot Control");
  ImGui::TextUnformatted("Formula: z(n+1) = z(n)^2 + C");
  ImGui::Separator();
  if (ImGui::Checkbox("Auto zoom (Space)", &state.autoZoom)) {
    state.lastInteractionSeconds = nowSeconds;
    state.viewRevision += 1;
    state.dirty = true;
  }
  ImGui::SliderFloat("Zoom speed", &state.zoomSpeed, 1.05f, 4.0f, "%.2f x/s");
  if (ImGui::SliderInt("Max iterations", &state.maxIterations, 64, kMaxIterationLimit)) {
    state.previewIterations = std::min(state.previewIterations, state.maxIterations);
    state.viewRevision += 1;
    state.dirty = true;
  }
  if (ImGui::SliderInt("Moving iterations", &state.previewIterations, 32, 20000)) {
    state.previewIterations = std::min(state.previewIterations, state.maxIterations);
    state.viewRevision += 1;
    state.dirty = true;
  }
  const int maxThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  state.cpuThreads = std::clamp(state.cpuThreads, 1, maxThreads);
  if (ImGui::SliderInt("CPU threads", &state.cpuThreads, 1, maxThreads)) {
    state.viewRevision += 1;
    state.dirty = true;
  }
  ImGui::Checkbox("Auto iterations", &state.autoIterations);
  if (ImGui::Checkbox("Smooth color", &state.smoothColor)) {
    state.viewRevision += 1;
    state.dirty = true;
  }
  const char* palettes[] = {"Cosine", "Heat", "Print"};
  if (ImGui::Combo("Palette", &state.palette, palettes, IM_ARRAYSIZE(palettes))) {
    state.viewRevision += 1;
    state.dirty = true;
  }
  if (ImGui::Button("Reset view (R)")) {
    resetView(state, nowSeconds);
  }
  ImGui::SameLine();
  if (ImGui::Button("Stop")) {
    state.autoZoom = false;
  }
  ImGui::Separator();
  ImGui::Text("Center X: %s", formatReal(state.centerX, 18).c_str());
  ImGui::Text("Center Y: %s", formatReal(state.centerY, 18).c_str());
  ImGui::Text("Zoom: %s", formatReal(state.zoom, 12).c_str());
  ImGui::Text("Frame: %dx%d, iterations %d", stats.displayedWidth, stats.displayedHeight, stats.displayedIterations);
  ImGui::Text("CPU render: %.2f ms", stats.renderMilliseconds);
  ImGui::Text("Renderer: %s, seq %llu", busy ? "working" : "idle", static_cast<unsigned long long>(stats.displayedSequence));
  char text[64] = {};
  std::snprintf(text, sizeof(text), "Progress %.1f%%", progress * 100.0f);
  ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), text);
  ImGui::Separator();
  ImGui::TextUnformatted("Left click: choose zoom target");
  ImGui::TextUnformatted("Left drag: pan");
  ImGui::TextUnformatted("Wheel: manual zoom");
  ImGui::TextUnformatted("Esc: quit");
  ImGui::End();
}

GLuint createOrUpdateTexture(GLuint texture, const RenderResult& frame) {
  if (texture == 0) {
    glGenTextures(1, &texture);
  }
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frame.width, frame.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame.pixels.data());
  return texture;
}

}  // namespace

int main() {
  if (!glfwInit()) {
    return 1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

  GLFWmonitor* monitor = glfwGetPrimaryMonitor();
  const GLFWvidmode* mode = monitor ? glfwGetVideoMode(monitor) : nullptr;
  const int initialWidth = mode ? std::max(960, static_cast<int>(mode->width * 0.85)) : 1440;
  const int initialHeight = mode ? std::max(720, static_cast<int>(mode->height * 0.85)) : 960;
  GLFWwindow* window = glfwCreateWindow(initialWidth, initialHeight, "Mandelbrot Realtime Zoom", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  glfwMaximizeWindow(window);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  const char* fontCandidates[] = {
      "/System/Library/Fonts/PingFang.ttc",
      "/System/Library/Fonts/STHeiti Light.ttc",
  };
  for (const char* fontPath : fontCandidates) {
    if (io.Fonts->AddFontFromFileTTF(fontPath, 18.0f, nullptr, io.Fonts->GetGlyphRangesChineseFull())) {
      break;
    }
  }
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 150");

  CpuRenderer renderer;
  AppState state;
  state.cpuThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  FrameStats stats;
  GLuint frameTexture = 0;
  uint64_t sequence = 0;
  bool wasMoving = false;
  auto lastTick = Clock::now();
  const auto startTick = lastTick;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    const auto now = Clock::now();
    const double dt = std::chrono::duration<double>(now - lastTick).count();
    const double nowSeconds = std::chrono::duration<double>(now - startTick).count();
    lastTick = now;

    int fbWidth = 1;
    int fbHeight = 1;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    fbWidth = std::max(1, fbWidth);
    fbHeight = std::max(1, fbHeight);

    updateAutoZoom(state, std::clamp(dt, 0.0, 0.1));

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    handleInput(state, fbWidth, fbHeight, nowSeconds);
    if (!io.WantCaptureKeyboard) {
      if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
        state.autoZoom = !state.autoZoom;
        state.lastInteractionSeconds = nowSeconds;
        state.viewRevision += 1;
        state.dirty = true;
      }
      if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
        resetView(state, nowSeconds);
      }
      if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
    }

    const bool moving = state.autoZoom || state.dragging || (nowSeconds - state.lastInteractionSeconds) < 0.20;
    if (wasMoving && !moving) {
      state.dirty = state.displayedViewRevision != state.viewRevision;
    }
    wasMoving = moving;

    if ((state.dirty || state.submittedViewRevision != state.viewRevision) && (!renderer.busy() || moving)) {
      RenderRequest request;
      const float scale = moving ? 0.5f : 1.0f;
      request.width = std::max(160, static_cast<int>(static_cast<float>(fbWidth) * scale));
      request.height = std::max(120, static_cast<int>(static_cast<float>(fbHeight) * scale));
      request.maxIterations = chooseIterations(state, moving);
      request.workerCount = state.cpuThreads;
      request.palette = state.palette;
      request.smoothColor = state.smoothColor;
      request.sequence = ++sequence;
      request.viewRevision = state.viewRevision;
      request.centerX = state.centerX;
      request.centerY = state.centerY;
      request.zoom = state.zoom;
      renderer.cancel(sequence);
      renderer.submit(request);
      state.submittedViewRevision = state.viewRevision;
      state.dirty = false;
    }

    RenderResult completed;
    if (renderer.takeResult(completed)) {
      if (completed.viewRevision == state.viewRevision) {
        frameTexture = createOrUpdateTexture(frameTexture, completed);
        stats.displayedWidth = completed.width;
        stats.displayedHeight = completed.height;
        stats.displayedIterations = completed.maxIterations;
        stats.renderMilliseconds = completed.milliseconds;
        stats.displayedSequence = completed.sequence;
        state.displayedViewRevision = completed.viewRevision;
      }
    }

    glViewport(0, 0, fbWidth, fbHeight);
    glClearColor(0.015f, 0.018f, 0.018f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    drawList->AddRectFilled(ImVec2(0, 0), ImVec2(static_cast<float>(fbWidth), static_cast<float>(fbHeight)), IM_COL32(4, 5, 5, 255));
    if (frameTexture != 0) {
      drawList->AddImage(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(frameTexture)),
                         ImVec2(0, 0),
                         ImVec2(static_cast<float>(fbWidth), static_cast<float>(fbHeight)));
    }

    drawControls(state, stats, renderer.busy(), renderer.progress(), nowSeconds);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  if (frameTexture != 0) {
    glDeleteTextures(1, &frameTexture);
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
