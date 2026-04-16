#include <windows.h>
#include <d3d11.h>
#include <quadmath.h>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"
#include "mandelbrot_compute_shader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace {

using Clock = std::chrono::steady_clock;
using Real = __float128;

constexpr int kRealDigits10 = 33;
constexpr int kRealMantissaBits = 113;
constexpr int kMaxIterationLimit = 100000000;
constexpr int kMaxPreviewIterationLimit = 20000;
constexpr int kMaxSeriesSkipLimit = 8192;
constexpr int kDynamicTileWidth = 32;
constexpr int kDynamicTileHeight = 8;
constexpr double kRecommendedFullFrameSeconds = 1.0;

Real realFromString(const char* text) {
  return strtoflt128(text, nullptr);
}

Real realAbs(Real value) {
  return fabsq(value);
}

Real realLog(Real value) {
  return logq(value);
}

Real realLog10(Real value) {
  return log10q(value);
}

Real realSqrt(Real value) {
  return sqrtq(value);
}

Real realPow(Real base, Real exponent) {
  return powq(base, exponent);
}

Real realExp(Real value) {
  return expq(value);
}

Real realNextAfter(Real value, Real target) {
  return nextafterq(value, target);
}

Real maxZoom() {
  static const Real value = realFromString("1e32");
  return value;
}

std::string formatReal(Real value, int precision = 18) {
  char buffer[160] = {};
  quadmath_snprintf(buffer, sizeof(buffer), "%.*Qe", precision, value);
  return buffer;
}

struct ComplexReal {
  Real x = 0.0L;
  Real y = 0.0L;
};

ComplexReal complexAdd(ComplexReal a, ComplexReal b) {
  return {a.x + b.x, a.y + b.y};
}

ComplexReal complexMul(ComplexReal a, ComplexReal b) {
  return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

ComplexReal complexScale(ComplexReal value, Real scale) {
  return {value.x * scale, value.y * scale};
}

Real complexAbs2(ComplexReal value) {
  return value.x * value.x + value.y * value.y;
}

Real complexAbs(ComplexReal value) {
  return realSqrt(complexAbs2(value));
}

ID3D11Device* g_device = nullptr;
ID3D11DeviceContext* g_deviceContext = nullptr;
IDXGISwapChain* g_swapChain = nullptr;
ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;
ID3D11Texture2D* g_frameTexture = nullptr;
ID3D11ShaderResourceView* g_frameTextureView = nullptr;
ID3D11UnorderedAccessView* g_frameTextureUav = nullptr;
ID3D11ComputeShader* g_mandelbrotComputeShader = nullptr;
ID3D11Buffer* g_computeParamsBuffer = nullptr;
int g_frameTextureWidth = 0;
int g_frameTextureHeight = 0;
UINT g_resizeWidth = 0;
UINT g_resizeHeight = 0;

struct GpuParams {
  float centerX;
  float centerY;
  float halfWidth;
  float halfHeight;
  int width;
  int height;
  int maxIterations;
  int palette;
};

struct PanCanvas {
  int viewWidth = 0;
  int viewHeight = 0;
  int canvasWidth = 0;
  int canvasHeight = 0;
  int originX = 0;
  int originY = 0;
  int maxIterations = 0;
  int samplesPerAxis = 1;
  int stage = 0;
  int palette = 0;
  bool smoothColor = true;
  Real baseCenterX = -0.5L;
  Real baseCenterY = 0.0L;
  Real zoom = 1.0L;
  std::vector<uint32_t> pixels;
  std::vector<uint8_t> known;
};

struct RenderRequest {
  int width = 1;
  int height = 1;
  int maxIterations = 256;
  int palette = 0;
  bool smoothColor = true;
  bool useSeriesApproximation = true;
  int samplesPerAxis = 1;
  int workerCount = 1;
  int stage = 0;
  int seriesMaxSkip = 2048;
  bool publishPatches = false;
  std::shared_ptr<const PanCanvas> panCanvas;
  uint64_t viewRevision = 0;
  Real centerX = -0.5L;
  Real centerY = 0.0L;
  Real zoom = 1.0L;
  uint64_t sequence = 0;
};

struct RenderResult {
  int width = 0;
  int height = 0;
  int maxIterations = 0;
  int samplesPerAxis = 1;
  int stage = 0;
  int seriesSkipIterations = 0;
  uint64_t sequence = 0;
  uint64_t viewRevision = 0;
  bool aborted = false;
  bool fromPanReuse = false;
  double milliseconds = 0.0;
  Real centerX = -0.5L;
  Real centerY = 0.0L;
  Real zoom = 1.0L;
  std::vector<uint32_t> pixels;
};

struct RenderPatch {
  int frameWidth = 0;
  int frameHeight = 0;
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
  int canvasX = 0;
  int canvasY = 0;
  uint64_t sequence = 0;
  uint64_t viewRevision = 0;
  std::vector<uint32_t> pixels;
};

struct AppState {
  Real centerX = -0.5L;
  Real centerY = 0.0L;
  Real targetX = -0.5L;
  Real targetY = 0.0L;
  Real zoom = 1.0L;
  int maxIterations = 5000;
  int previewIterationCap = 500;
  int supersample = 1;
  int cpuThreads = 1;
  int seriesMaxSkip = 2048;
  int palette = 0;
  float movingScale = 1.0f;
  float stillScale = 1.0f;
  float zoomSpeed = 1.75f;
  bool gpuPreview = true;
  bool autoZoom = false;
  bool autoIterations = true;
  bool smoothColor = true;
  bool useSeriesApproximation = true;
  bool mouseDown = false;
  bool dragging = false;
  ImVec2 mouseDownPos = ImVec2(0.0f, 0.0f);
  Real dragCenterX = -0.5L;
  Real dragCenterY = 0.0L;
  double lastDragSeconds = -1000.0;
  double lastWheelZoomSeconds = -1000.0;
  double lastInteractionSeconds = 0.0;
  double lastRenderSubmitSeconds = -1.0;
  uint64_t viewRevision = 1;
  uint64_t submittedViewRevision = 0;
  int submittedStage = -1;
  uint64_t displayedViewRevision = 0;
  int displayedStage = -1;
  bool textureFrameValid = false;
  int textureFrameWidth = 0;
  int textureFrameHeight = 0;
  Real textureFrameCenterX = -0.5L;
  Real textureFrameCenterY = 0.0L;
  Real textureFrameZoom = 1.0L;
  bool dirty = true;
};

struct FrameStats {
  int displayedWidth = 0;
  int displayedHeight = 0;
  int displayedIterations = 0;
  int displayedSamplesPerAxis = 1;
  int displayedStage = -1;
  int displayedSeriesSkipIterations = 0;
  double renderMilliseconds = 0.0;
  uint64_t displayedSequence = 0;
  bool displayedGpuPreview = false;
  bool displayedPanReuse = false;
  bool gpuPreviewPrecisionOk = true;
  double estimatedIterationsPerSecond = 0.0;
};

struct SeriesApproximation {
  bool enabled = false;
  int skipIterations = 0;
  Real referenceX = -0.5L;
  Real referenceY = 0.0L;
  ComplexReal referenceZ;
  ComplexReal linear;
  ComplexReal quadratic;
  ComplexReal cubic;
};

SeriesApproximation buildSeriesApproximation(const RenderRequest& request, Real halfWidth, Real halfHeight) {
  SeriesApproximation result;
  result.referenceX = request.centerX;
  result.referenceY = request.centerY;

  if (!request.useSeriesApproximation || request.maxIterations < 256 || request.zoom < 32.0L) {
    return result;
  }

  const Real radius = realSqrt(halfWidth * halfWidth + halfHeight * halfHeight);
  if (radius <= 0.0L || radius > 0.125L) {
    return result;
  }

  const int maxSkip = std::clamp(
      std::min(request.seriesMaxSkip, request.maxIterations - 1),
      0,
      kMaxSeriesSkipLimit);
  if (maxSkip < 16) {
    return result;
  }

  ComplexReal z;
  ComplexReal a;
  ComplexReal b;
  ComplexReal c;
  const ComplexReal center = {request.centerX, request.centerY};
  const Real radius2 = radius * radius;
  const Real radius3 = radius2 * radius;
  const Real tiny = realFromString("1e-36");

  for (int iter = 1; iter <= maxSkip; ++iter) {
    if (complexAbs2(z) > 4.0L) {
      break;
    }

    const ComplexReal oldZ = z;
    const ComplexReal oldA = a;
    const ComplexReal oldB = b;
    const ComplexReal oldC = c;

    z = complexAdd(complexMul(oldZ, oldZ), center);
    a = complexAdd(complexScale(complexMul(oldZ, oldA), 2.0L), {1.0L, 0.0L});
    b = complexAdd(complexMul(oldA, oldA), complexScale(complexMul(oldZ, oldB), 2.0L));
    c = complexAdd(complexScale(complexMul(oldA, oldB), 2.0L), complexScale(complexMul(oldZ, oldC), 2.0L));

    const Real term1 = complexAbs(a) * radius;
    const Real term2 = complexAbs(b) * radius2;
    const Real term3 = complexAbs(c) * radius3;
    const Real spread = term1 + term2 + term3;
    const bool termsControlled =
        term2 <= std::max(term1 * 0.50L, tiny) &&
        term3 <= std::max(term2 * 0.25L, tiny);
    const bool orbitControlled = complexAbs2(z) < 16.0L && spread < 1.50L;

    if (iter >= 16 && termsControlled && orbitControlled) {
      result.enabled = true;
      result.skipIterations = iter;
      result.referenceZ = z;
      result.linear = a;
      result.quadratic = b;
      result.cubic = c;
    }
  }

  return result;
}

bool tryStartFromSeries(
    const SeriesApproximation& series,
    Real cx,
    Real cy,
    Real& zx,
    Real& zy,
    Real& zx2,
    Real& zy2,
    int& iter) {
  if (!series.enabled || series.skipIterations <= 0) {
    return false;
  }

  const ComplexReal dc = {cx - series.referenceX, cy - series.referenceY};
  const ComplexReal dc2 = complexMul(dc, dc);
  const ComplexReal dc3 = complexMul(dc2, dc);
  ComplexReal z = series.referenceZ;
  z = complexAdd(z, complexMul(series.linear, dc));
  z = complexAdd(z, complexMul(series.quadratic, dc2));
  z = complexAdd(z, complexMul(series.cubic, dc3));

  const Real zAbs2 = complexAbs2(z);
  if (zAbs2 > 4.0L) {
    return false;
  }

  zx = z.x;
  zy = z.y;
  zx2 = zx * zx;
  zy2 = zy * zy;
  iter = series.skipIterations;
  return true;
}

uint8_t toByte(double value) {
  value = std::clamp(value, 0.0, 255.0);
  return static_cast<uint8_t>(value + 0.5);
}

uint32_t packBgra(uint8_t r, uint8_t g, uint8_t b) {
  return 0xff000000u | (static_cast<uint32_t>(r) << 16u) |
         (static_cast<uint32_t>(g) << 8u) | static_cast<uint32_t>(b);
}

uint32_t paletteColor(double t, int palette) {
  t = std::clamp(t, 0.0, 1.0);
  const double pi2 = 6.2831853071795864769;

  if (palette == 1) {
    const double r = 40.0 + 215.0 * std::pow(t, 0.38);
    const double g = 20.0 + 190.0 * std::sin(t * 1.57079632679);
    const double b = 95.0 + 120.0 * (1.0 - t);
    return packBgra(toByte(r), toByte(g), toByte(b));
  }

  if (palette == 2) {
    const double band = 0.5 + 0.5 * std::cos(pi2 * (t * 12.0));
    const double base = 255.0 * std::pow(t, 0.22);
    return packBgra(toByte(base), toByte(base * (0.72 + 0.28 * band)), toByte(base * 0.78));
  }

  if (palette == 3) {
    const double u = t * 18.0;
    const int band = static_cast<int>(std::floor(u));
    const double local = u - static_cast<double>(band);
    const double hard = local < 0.58 ? 1.0 : 0.05;
    const uint8_t colors[][3] = {
        {255, 245, 20},
        {0, 235, 255},
        {255, 0, 220},
        {255, 255, 255},
        {40, 255, 0},
        {255, 80, 0},
    };
    const uint8_t* c = colors[band % 6];
    return packBgra(toByte(c[0] * hard), toByte(c[1] * hard), toByte(c[2] * hard));
  }

  const double r = 128.0 + 127.0 * std::cos(pi2 * (t + 0.04));
  const double g = 128.0 + 127.0 * std::cos(pi2 * (t + 0.34));
  const double b = 128.0 + 127.0 * std::cos(pi2 * (t + 0.67));
  return packBgra(toByte(r), toByte(g), toByte(b));
}

uint32_t mandelbrotPixel(
    Real cx,
    Real cy,
    int maxIterations,
    int palette,
    bool smoothColor,
    const SeriesApproximation& series,
    const std::atomic<uint64_t>& latestSequence,
    uint64_t sequence,
    bool& canceled) {
  const Real xMinusQuarter = cx - 0.25L;
  const Real q = xMinusQuarter * xMinusQuarter + cy * cy;
  if (q * (q + xMinusQuarter) <= 0.25L * cy * cy) {
    return packBgra(4, 5, 5);
  }

  const Real xPlusOne = cx + 1.0L;
  if (xPlusOne * xPlusOne + cy * cy <= 0.0625L) {
    return packBgra(4, 5, 5);
  }

  Real zx = 0.0L;
  Real zy = 0.0L;
  Real zx2 = 0.0L;
  Real zy2 = 0.0L;
  int iter = 0;
  tryStartFromSeries(series, cx, cy, zx, zy, zx2, zy2, iter);

  for (; iter < maxIterations; ++iter) {
    if ((iter & 1023) == 0 && latestSequence.load(std::memory_order_relaxed) != sequence) {
      canceled = true;
      return packBgra(4, 5, 5);
    }

    if (zx2 + zy2 > 4.0L) {
      break;
    }

    zy = 2.0L * zx * zy + cy;
    zx = zx2 - zy2 + cx;
    zx2 = zx * zx;
    zy2 = zy * zy;
  }

  if (iter >= maxIterations) {
    return packBgra(4, 5, 5);
  }

  Real value = static_cast<Real>(iter);
  if (smoothColor && iter > 0) {
    const Real magnitude2 = std::max(zx2 + zy2, static_cast<Real>(4.0000000001L));
    const Real logZn = 0.5L * realLog(magnitude2);
    const Real log2 = realLog(2.0L);
    const Real nu = realLog(logZn / log2) / log2;
    value = static_cast<Real>(iter) + 1.0L - nu;
  }

  const double t = static_cast<double>(realSqrt(std::max(value, static_cast<Real>(0.0L)) / static_cast<Real>(maxIterations)));
  return paletteColor(t, palette);
}

int colorDistance(uint32_t a, uint32_t b) {
  const int ar = static_cast<int>((a >> 16u) & 0xffu);
  const int ag = static_cast<int>((a >> 8u) & 0xffu);
  const int ab = static_cast<int>(a & 0xffu);
  const int br = static_cast<int>((b >> 16u) & 0xffu);
  const int bg = static_cast<int>((b >> 8u) & 0xffu);
  const int bb = static_cast<int>(b & 0xffu);
  return std::max({std::abs(ar - br), std::abs(ag - bg), std::abs(ab - bb)});
}

int adaptiveTileSize(int stage) {
  if (stage < 0 || stage >= 3) {
    return 1;
  }
  return 4;
}

int adaptiveTileTolerance(int stage) {
  if (stage <= 0) {
    return 48;
  }
  if (stage == 1) {
    return 28;
  }
  return 14;
}

int renderTileWidth(const RenderRequest& request) {
  const int adaptiveSize = adaptiveTileSize(request.stage);
  if (adaptiveSize > 1 && request.samplesPerAxis == 1) {
    return adaptiveSize;
  }
  return kDynamicTileWidth;
}

int renderTileHeight(const RenderRequest& request) {
  const int adaptiveSize = adaptiveTileSize(request.stage);
  if (adaptiveSize > 1 && request.samplesPerAxis == 1) {
    return adaptiveSize;
  }
  return kDynamicTileHeight;
}

int renderTileCount(const RenderRequest& request) {
  const int tileWidth = std::max(1, renderTileWidth(request));
  const int tileHeight = std::max(1, renderTileHeight(request));
  const int tileCols = (std::max(1, request.width) + tileWidth - 1) / tileWidth;
  const int tileRows = (std::max(1, request.height) + tileHeight - 1) / tileHeight;
  return std::max(1, tileCols * tileRows);
}

uint32_t averagePixelColor(
    int x,
    int y,
    const RenderRequest& request,
    const std::vector<Real>& xCoords,
    const std::vector<Real>& yCoords,
    Real pixelStepX,
    Real pixelStepY,
    const SeriesApproximation& series,
    const std::atomic<uint64_t>& latestSequence,
    bool& canceled) {
  const int samplesPerAxis = std::clamp(request.samplesPerAxis, 1, 4);
  if (samplesPerAxis == 1) {
    const Real cx = xCoords[static_cast<size_t>(x)];
    const Real cy = yCoords[static_cast<size_t>(y)];
    return mandelbrotPixel(
        cx,
        cy,
        request.maxIterations,
        request.palette,
        request.smoothColor,
        series,
        latestSequence,
        request.sequence,
        canceled);
  }

  uint32_t rSum = 0;
  uint32_t gSum = 0;
  uint32_t bSum = 0;
  const int sampleCount = samplesPerAxis * samplesPerAxis;
  const Real invSamples = 1.0L / static_cast<Real>(samplesPerAxis);

  for (int sy = 0; sy < samplesPerAxis; ++sy) {
    for (int sx = 0; sx < samplesPerAxis; ++sx) {
      const Real ox = (static_cast<Real>(sx) + 0.5L) * invSamples;
      const Real oy = (static_cast<Real>(sy) + 0.5L) * invSamples;
      const Real cx = xCoords[static_cast<size_t>(x)] + (ox - 0.5L) * pixelStepX;
      const Real cy = yCoords[static_cast<size_t>(y)] - (oy - 0.5L) * pixelStepY;
      const uint32_t color = mandelbrotPixel(
          cx,
          cy,
          request.maxIterations,
          request.palette,
          request.smoothColor,
          series,
          latestSequence,
          request.sequence,
          canceled);
      if (canceled) {
        return packBgra(4, 5, 5);
      }
      bSum += color & 0xffu;
      gSum += (color >> 8u) & 0xffu;
      rSum += (color >> 16u) & 0xffu;
    }
  }

  const uint8_t r = static_cast<uint8_t>(rSum / static_cast<uint32_t>(sampleCount));
  const uint8_t g = static_cast<uint8_t>(gSum / static_cast<uint32_t>(sampleCount));
  const uint8_t b = static_cast<uint8_t>(bSum / static_cast<uint32_t>(sampleCount));
  return packBgra(r, g, b);
}

bool tryFillAdaptiveTile(
    int x0,
    int y0,
    int x1,
    int y1,
    const RenderRequest& request,
    const std::vector<Real>& xCoords,
    const std::vector<Real>& yCoords,
    Real pixelStepX,
    Real pixelStepY,
    const SeriesApproximation& series,
    const std::atomic<uint64_t>& latestSequence,
    uint32_t* pixels,
    bool& canceled) {
  const int tileWidth = x1 - x0;
  const int tileHeight = y1 - y0;
  if (tileWidth <= 1 || tileHeight <= 1 || request.samplesPerAxis != 1) {
    return false;
  }

  const int cx = (x0 + x1 - 1) / 2;
  const int cy = (y0 + y1 - 1) / 2;
  const uint32_t center = averagePixelColor(
      cx, cy, request, xCoords, yCoords, pixelStepX, pixelStepY, series, latestSequence, canceled);
  if (canceled) {
    return false;
  }

  const int tolerance = adaptiveTileTolerance(request.stage);
  const int sampleXs[4] = {x0, x1 - 1, x0, x1 - 1};
  const int sampleYs[4] = {y0, y0, y1 - 1, y1 - 1};
  for (int i = 0; i < 4; ++i) {
    const uint32_t color = averagePixelColor(
        sampleXs[i],
        sampleYs[i],
        request,
        xCoords,
        yCoords,
        pixelStepX,
        pixelStepY,
        series,
        latestSequence,
        canceled);
    if (canceled) {
      return false;
    }
    if (colorDistance(color, center) > tolerance) {
      return false;
    }
  }

  for (int y = y0; y < y1; ++y) {
    uint32_t* row = pixels + static_cast<size_t>(y) * static_cast<size_t>(request.width);
    std::fill(row + x0, row + x1, center);
  }
  return true;
}

bool panCanvasReady(const PanCanvas& canvas) {
  return canvas.viewWidth > 0 && canvas.viewHeight > 0 &&
         canvas.canvasWidth > 0 && canvas.canvasHeight > 0 &&
         canvas.pixels.size() == static_cast<size_t>(canvas.canvasWidth) * static_cast<size_t>(canvas.canvasHeight) &&
         canvas.known.size() == canvas.pixels.size();
}

Real panPixelStepX(const PanCanvas& canvas) {
  const Real aspect = static_cast<Real>(canvas.viewWidth) / static_cast<Real>(std::max(1, canvas.viewHeight));
  return (3.0L / canvas.zoom * aspect) / static_cast<Real>(std::max(1, canvas.viewWidth));
}

Real panPixelStepY(const PanCanvas& canvas) {
  return (3.0L / canvas.zoom) / static_cast<Real>(std::max(1, canvas.viewHeight));
}

int panViewLeft(const PanCanvas& canvas, Real centerX) {
  return static_cast<int>(std::llround(static_cast<double>((centerX - canvas.baseCenterX) / panPixelStepX(canvas))));
}

int panViewTop(const PanCanvas& canvas, Real centerY) {
  return static_cast<int>(std::llround(static_cast<double>((canvas.baseCenterY - centerY) / panPixelStepY(canvas))));
}

bool panCanvasCompatible(const PanCanvas& canvas, const RenderRequest& request) {
  return panCanvasReady(canvas) &&
         canvas.viewWidth == request.width &&
         canvas.viewHeight == request.height &&
         canvas.zoom == request.zoom &&
         canvas.maxIterations == request.maxIterations &&
         canvas.samplesPerAxis == request.samplesPerAxis &&
         canvas.palette == request.palette &&
         canvas.smoothColor == request.smoothColor;
}

bool panCanvasLookup(const PanCanvas& canvas, int globalX, int globalY, uint32_t& color) {
  const int localX = globalX - canvas.originX;
  const int localY = globalY - canvas.originY;
  if (localX < 0 || localY < 0 || localX >= canvas.canvasWidth || localY >= canvas.canvasHeight) {
    return false;
  }

  const size_t index = static_cast<size_t>(localY) * static_cast<size_t>(canvas.canvasWidth) +
                       static_cast<size_t>(localX);
  if (!canvas.known[index]) {
    return false;
  }

  color = canvas.pixels[index];
  return true;
}

RenderResult cropPanCanvasView(
    const PanCanvas& canvas,
    Real centerX,
    Real centerY,
    uint64_t viewRevision,
    uint64_t sequence) {
  RenderResult result;
  result.width = canvas.viewWidth;
  result.height = canvas.viewHeight;
  result.maxIterations = canvas.maxIterations;
  result.samplesPerAxis = canvas.samplesPerAxis;
  result.stage = 0;
  result.sequence = sequence;
  result.viewRevision = viewRevision;
  result.centerX = centerX;
  result.centerY = centerY;
  result.zoom = canvas.zoom;
  result.fromPanReuse = true;
  result.pixels.assign(
      static_cast<size_t>(result.width) * static_cast<size_t>(result.height),
      packBgra(4, 5, 5));

  const int viewLeft = panViewLeft(canvas, centerX);
  const int viewTop = panViewTop(canvas, centerY);
  for (int y = 0; y < result.height; ++y) {
    uint32_t* row = result.pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(result.width);
    const int globalY = viewTop + y;
    for (int x = 0; x < result.width; ++x) {
      uint32_t color = 0;
      if (panCanvasLookup(canvas, viewLeft + x, globalY, color)) {
        row[x] = color;
      }
    }
  }

  return result;
}

RenderPatch makeRenderPatch(
    const RenderResult& frame,
    int x0,
    int y0,
    int x1,
    int y1,
    int canvasX,
    int canvasY) {
  RenderPatch patch;
  patch.frameWidth = frame.width;
  patch.frameHeight = frame.height;
  patch.x = x0;
  patch.y = y0;
  patch.width = x1 - x0;
  patch.height = y1 - y0;
  patch.canvasX = canvasX;
  patch.canvasY = canvasY;
  patch.sequence = frame.sequence;
  patch.viewRevision = frame.viewRevision;
  if (patch.width <= 0 || patch.height <= 0) {
    return patch;
  }

  patch.pixels.resize(static_cast<size_t>(patch.width) * static_cast<size_t>(patch.height));
  for (int row = 0; row < patch.height; ++row) {
    const size_t src = static_cast<size_t>(y0 + row) * static_cast<size_t>(frame.width) +
                       static_cast<size_t>(x0);
    const size_t dst = static_cast<size_t>(row) * static_cast<size_t>(patch.width);
    std::copy(frame.pixels.begin() + static_cast<std::ptrdiff_t>(src),
              frame.pixels.begin() + static_cast<std::ptrdiff_t>(src + patch.width),
              patch.pixels.begin() + static_cast<std::ptrdiff_t>(dst));
  }
  return patch;
}

RenderResult renderPanCanvasMandelbrot(
    const RenderRequest& request,
    const PanCanvas& canvas,
    const std::atomic<uint64_t>& latestSequence,
    std::atomic<int>& progressDone,
    const std::function<void(RenderPatch&&)>& publishPatch) {
  RenderResult result;
  result.width = request.width;
  result.height = request.height;
  result.maxIterations = request.maxIterations;
  result.samplesPerAxis = request.samplesPerAxis;
  result.stage = 0;
  result.sequence = request.sequence;
  result.viewRevision = request.viewRevision;
  result.centerX = request.centerX;
  result.centerY = request.centerY;
  result.zoom = request.zoom;
  result.fromPanReuse = true;
  result.pixels.assign(
      static_cast<size_t>(request.width) * static_cast<size_t>(request.height),
      packBgra(4, 5, 5));

  const auto start = Clock::now();
  const Real aspect = static_cast<Real>(request.width) / static_cast<Real>(request.height);
  const Real halfHeight = 1.5L / request.zoom;
  const Real halfWidth = halfHeight * aspect;
  const SeriesApproximation series = buildSeriesApproximation(request, halfWidth, halfHeight);
  result.seriesSkipIterations = series.skipIterations;
  const Real pixelStepX = (2.0L * halfWidth) / static_cast<Real>(request.width);
  const Real pixelStepY = (2.0L * halfHeight) / static_cast<Real>(request.height);
  const Real left = request.centerX - halfWidth;
  const Real top = request.centerY + halfHeight;
  std::vector<Real> xCoords(static_cast<size_t>(request.width));
  std::vector<Real> yCoords(static_cast<size_t>(request.height));
  Real cx = left + 0.5L * pixelStepX;
  for (int x = 0; x < request.width; ++x) {
    xCoords[static_cast<size_t>(x)] = cx;
    cx += pixelStepX;
  }
  Real cy = top - 0.5L * pixelStepY;
  for (int y = 0; y < request.height; ++y) {
    yCoords[static_cast<size_t>(y)] = cy;
    cy -= pixelStepY;
  }

  auto isCanceled = [&]() {
    return latestSequence.load(std::memory_order_relaxed) != request.sequence;
  };

  const int viewLeft = panViewLeft(canvas, request.centerX);
  const int viewTop = panViewTop(canvas, request.centerY);
  const int workerCount = std::clamp(request.workerCount, 1, 64);
  const int tileWidth = kDynamicTileWidth;
  const int tileHeight = kDynamicTileHeight;
  const int tileCols = (request.width + tileWidth - 1) / tileWidth;
  const int tileRows = (request.height + tileHeight - 1) / tileHeight;
  const int totalTiles = std::max(1, tileCols * tileRows);
  std::atomic<int> nextTile = 0;
  std::atomic<bool> canceled = false;
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(workerCount));

  auto renderTileQueue = [&]() {
    for (;;) {
      if (isCanceled()) {
        canceled.store(true, std::memory_order_relaxed);
        return;
      }

      const int tileIndex = nextTile.fetch_add(1, std::memory_order_relaxed);
      if (tileIndex >= totalTiles) {
        return;
      }

      const int tileX = tileIndex % tileCols;
      const int tileY = tileIndex / tileCols;
      const int x0 = tileX * tileWidth;
      const int y0 = tileY * tileHeight;
      const int x1 = std::min(request.width, x0 + tileWidth);
      const int y1 = std::min(request.height, y0 + tileHeight);

      for (int y = y0; y < y1; ++y) {
        uint32_t* row = result.pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(request.width);
        const int globalY = viewTop + y;
        for (int x = x0; x < x1; ++x) {
          uint32_t color = 0;
          if (panCanvasLookup(canvas, viewLeft + x, globalY, color)) {
            row[x] = color;
            continue;
          }
          if ((x & 31) == 0 && isCanceled()) {
            canceled.store(true, std::memory_order_relaxed);
            return;
          }

          bool pixelCanceled = false;
          row[x] = averagePixelColor(
              x, y, request, xCoords, yCoords, pixelStepX, pixelStepY, series, latestSequence, pixelCanceled);
          if (pixelCanceled) {
            canceled.store(true, std::memory_order_relaxed);
            return;
          }
        }
      }

      if (request.publishPatches && publishPatch) {
        publishPatch(makeRenderPatch(result, x0, y0, x1, y1, viewLeft + x0, viewTop + y0));
      }
      progressDone.fetch_add(1, std::memory_order_relaxed);
    }
  };

  for (int i = 0; i < workerCount; ++i) {
    workers.emplace_back(renderTileQueue);
  }
  for (std::thread& worker : workers) {
    worker.join();
  }

  const auto end = Clock::now();
  result.aborted = canceled.load(std::memory_order_relaxed) || isCanceled();
  result.milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
  return result;
}

RenderResult renderMandelbrot(
    const RenderRequest& request,
    const std::atomic<uint64_t>& latestSequence,
    std::atomic<int>& progressDone,
    const std::function<void(RenderPatch&&)>& publishPatch = {}) {
  if (request.panCanvas && panCanvasCompatible(*request.panCanvas, request)) {
    return renderPanCanvasMandelbrot(request, *request.panCanvas, latestSequence, progressDone, publishPatch);
  }

  RenderResult result;
  result.width = request.width;
  result.height = request.height;
  result.maxIterations = request.maxIterations;
  result.samplesPerAxis = request.samplesPerAxis;
  result.stage = request.stage;
  result.sequence = request.sequence;
  result.viewRevision = request.viewRevision;
  result.centerX = request.centerX;
  result.centerY = request.centerY;
  result.zoom = request.zoom;
  result.pixels.resize(static_cast<size_t>(request.width) * static_cast<size_t>(request.height));

  const auto start = Clock::now();
  const int workerCount = std::clamp(request.workerCount, 1, 64);
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(workerCount));

  const Real aspect = static_cast<Real>(request.width) / static_cast<Real>(request.height);
  const Real halfHeight = 1.5L / request.zoom;
  const Real halfWidth = halfHeight * aspect;
  const SeriesApproximation series = buildSeriesApproximation(request, halfWidth, halfHeight);
  result.seriesSkipIterations = series.skipIterations;
  const Real pixelStepX = (2.0L * halfWidth) / static_cast<Real>(request.width);
  const Real pixelStepY = (2.0L * halfHeight) / static_cast<Real>(request.height);
  const Real left = request.centerX - halfWidth;
  const Real top = request.centerY + halfHeight;
  std::vector<Real> xCoords(static_cast<size_t>(request.width));
  std::vector<Real> yCoords(static_cast<size_t>(request.height));
  Real cx = left + 0.5L * pixelStepX;
  for (int x = 0; x < request.width; ++x) {
    xCoords[static_cast<size_t>(x)] = cx;
    cx += pixelStepX;
  }
  Real cy = top - 0.5L * pixelStepY;
  for (int y = 0; y < request.height; ++y) {
    yCoords[static_cast<size_t>(y)] = cy;
    cy -= pixelStepY;
  }

  const int tileSize = adaptiveTileSize(request.stage);
  const bool useAdaptiveTiles = tileSize > 1 && request.samplesPerAxis == 1;
  const int workTileWidth = renderTileWidth(request);
  const int workTileHeight = renderTileHeight(request);
  const int tileCols = (request.width + workTileWidth - 1) / workTileWidth;
  const int tileRows = (request.height + workTileHeight - 1) / workTileHeight;
  const int totalTiles = std::max(1, tileCols * tileRows);
  std::atomic<int> nextTile = 0;
  std::atomic<bool> canceled = false;

  auto isCanceled = [&]() {
    return latestSequence.load(std::memory_order_relaxed) != request.sequence;
  };

  auto renderTileQueue = [&]() {
    for (;;) {
      if (isCanceled()) {
        canceled.store(true, std::memory_order_relaxed);
        return;
      }

      const int tileIndex = nextTile.fetch_add(1, std::memory_order_relaxed);
      if (tileIndex >= totalTiles) {
        return;
      }

      const int tileX = tileIndex % tileCols;
      const int tileY = tileIndex / tileCols;
      const int x0 = tileX * workTileWidth;
      const int y0 = tileY * workTileHeight;
      const int x1 = std::min(request.width, x0 + workTileWidth);
      const int y1 = std::min(request.height, y0 + workTileHeight);

      if (useAdaptiveTiles) {
        if (isCanceled()) {
          canceled.store(true, std::memory_order_relaxed);
          return;
        }

        bool tileCanceled = false;
        if (tryFillAdaptiveTile(
                x0,
                y0,
                x1,
                y1,
                request,
                xCoords,
                yCoords,
                pixelStepX,
                pixelStepY,
                series,
                latestSequence,
                result.pixels.data(),
                tileCanceled)) {
          continue;
        }
        if (tileCanceled) {
          canceled.store(true, std::memory_order_relaxed);
          return;
        }
      }

      for (int y = y0; y < y1; ++y) {
        uint32_t* row = result.pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(request.width);
        for (int x = x0; x < x1; ++x) {
          if ((x & 31) == 0 && isCanceled()) {
            canceled.store(true, std::memory_order_relaxed);
            return;
          }
          bool pixelCanceled = false;
          row[x] = averagePixelColor(
              x, y, request, xCoords, yCoords, pixelStepX, pixelStepY, series, latestSequence, pixelCanceled);
          if (pixelCanceled) {
            canceled.store(true, std::memory_order_relaxed);
            return;
          }
        }
      }

      if (request.publishPatches && publishPatch) {
        publishPatch(makeRenderPatch(result, x0, y0, x1, y1, x0, y0));
      }
      progressDone.fetch_add(1, std::memory_order_relaxed);
    }
  };

  for (int i = 0; i < workerCount; ++i) {
    workers.emplace_back(renderTileQueue);
  }

  for (std::thread& worker : workers) {
    worker.join();
  }

  const auto end = Clock::now();
  result.aborted = canceled.load(std::memory_order_relaxed) || isCanceled();
  result.milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
  return result;
}

std::shared_ptr<PanCanvas> makePanCanvasFromFrame(const RenderResult& frame, int palette, bool smoothColor) {
  if (frame.width <= 0 || frame.height <= 0 || frame.pixels.empty()) {
    return nullptr;
  }

  auto canvas = std::make_shared<PanCanvas>();
  canvas->viewWidth = frame.width;
  canvas->viewHeight = frame.height;
  canvas->canvasWidth = frame.width;
  canvas->canvasHeight = frame.height;
  canvas->originX = 0;
  canvas->originY = 0;
  canvas->maxIterations = frame.maxIterations;
  canvas->samplesPerAxis = frame.samplesPerAxis;
  canvas->stage = frame.stage;
  canvas->palette = palette;
  canvas->smoothColor = smoothColor;
  canvas->baseCenterX = frame.centerX;
  canvas->baseCenterY = frame.centerY;
  canvas->zoom = frame.zoom;
  canvas->pixels = frame.pixels;
  canvas->known.assign(canvas->pixels.size(), 1);
  return canvas;
}

bool ensurePanCanvasCovers(PanCanvas& canvas, int viewLeft, int viewTop, int viewWidth, int viewHeight) {
  if (!panCanvasReady(canvas)) {
    return false;
  }

  const int oldRight = canvas.originX + canvas.canvasWidth;
  const int oldBottom = canvas.originY + canvas.canvasHeight;
  const int viewRight = viewLeft + viewWidth;
  const int viewBottom = viewTop + viewHeight;
  const int newOriginX = std::min(canvas.originX, viewLeft);
  const int newOriginY = std::min(canvas.originY, viewTop);
  const int newRight = std::max(oldRight, viewRight);
  const int newBottom = std::max(oldBottom, viewBottom);
  const int newWidth = newRight - newOriginX;
  const int newHeight = newBottom - newOriginY;

  if (newWidth == canvas.canvasWidth && newHeight == canvas.canvasHeight &&
      newOriginX == canvas.originX && newOriginY == canvas.originY) {
    return true;
  }
  if (newWidth <= 0 || newHeight <= 0) {
    return false;
  }

  const size_t newSize = static_cast<size_t>(newWidth) * static_cast<size_t>(newHeight);
  std::vector<uint32_t> newPixels(newSize, packBgra(4, 5, 5));
  std::vector<uint8_t> newKnown(newSize, 0);
  const int dstOffsetX = canvas.originX - newOriginX;
  const int dstOffsetY = canvas.originY - newOriginY;

  for (int y = 0; y < canvas.canvasHeight; ++y) {
    const size_t oldRow = static_cast<size_t>(y) * static_cast<size_t>(canvas.canvasWidth);
    const size_t newRow = static_cast<size_t>(y + dstOffsetY) * static_cast<size_t>(newWidth) +
                          static_cast<size_t>(dstOffsetX);
    std::copy(canvas.pixels.begin() + static_cast<std::ptrdiff_t>(oldRow),
              canvas.pixels.begin() + static_cast<std::ptrdiff_t>(oldRow + canvas.canvasWidth),
              newPixels.begin() + static_cast<std::ptrdiff_t>(newRow));
    std::copy(canvas.known.begin() + static_cast<std::ptrdiff_t>(oldRow),
              canvas.known.begin() + static_cast<std::ptrdiff_t>(oldRow + canvas.canvasWidth),
              newKnown.begin() + static_cast<std::ptrdiff_t>(newRow));
  }

  canvas.originX = newOriginX;
  canvas.originY = newOriginY;
  canvas.canvasWidth = newWidth;
  canvas.canvasHeight = newHeight;
  canvas.pixels = std::move(newPixels);
  canvas.known = std::move(newKnown);
  return true;
}

bool mergeFrameIntoPanCanvas(PanCanvas& canvas, const RenderResult& frame) {
  if (!panCanvasReady(canvas) || frame.width != canvas.viewWidth || frame.height != canvas.viewHeight ||
      frame.zoom != canvas.zoom || frame.maxIterations != canvas.maxIterations ||
      frame.samplesPerAxis != canvas.samplesPerAxis || frame.pixels.empty()) {
    return false;
  }

  const int viewLeft = panViewLeft(canvas, frame.centerX);
  const int viewTop = panViewTop(canvas, frame.centerY);
  if (!ensurePanCanvasCovers(canvas, viewLeft, viewTop, frame.width, frame.height)) {
    return false;
  }

  const int localX = viewLeft - canvas.originX;
  const int localY = viewTop - canvas.originY;
  for (int y = 0; y < frame.height; ++y) {
    const size_t srcRow = static_cast<size_t>(y) * static_cast<size_t>(frame.width);
    const size_t dstRow = static_cast<size_t>(localY + y) * static_cast<size_t>(canvas.canvasWidth) +
                          static_cast<size_t>(localX);
    std::copy(frame.pixels.begin() + static_cast<std::ptrdiff_t>(srcRow),
              frame.pixels.begin() + static_cast<std::ptrdiff_t>(srcRow + frame.width),
              canvas.pixels.begin() + static_cast<std::ptrdiff_t>(dstRow));
    std::fill(canvas.known.begin() + static_cast<std::ptrdiff_t>(dstRow),
              canvas.known.begin() + static_cast<std::ptrdiff_t>(dstRow + frame.width),
              static_cast<uint8_t>(1));
  }
  return true;
}

bool mergePatchIntoPanCanvas(PanCanvas& canvas, const RenderPatch& patch) {
  if (!panCanvasReady(canvas) || patch.width <= 0 || patch.height <= 0 ||
      patch.pixels.size() != static_cast<size_t>(patch.width) * static_cast<size_t>(patch.height)) {
    return false;
  }

  if (!ensurePanCanvasCovers(canvas, patch.canvasX, patch.canvasY, patch.width, patch.height)) {
    return false;
  }

  const int localX = patch.canvasX - canvas.originX;
  const int localY = patch.canvasY - canvas.originY;
  if (localX < 0 || localY < 0 ||
      localX + patch.width > canvas.canvasWidth ||
      localY + patch.height > canvas.canvasHeight) {
    return false;
  }
  for (int y = 0; y < patch.height; ++y) {
    const size_t srcRow = static_cast<size_t>(y) * static_cast<size_t>(patch.width);
    const size_t dstRow = static_cast<size_t>(localY + y) * static_cast<size_t>(canvas.canvasWidth) +
                          static_cast<size_t>(localX);
    std::copy(patch.pixels.begin() + static_cast<std::ptrdiff_t>(srcRow),
              patch.pixels.begin() + static_cast<std::ptrdiff_t>(srcRow + patch.width),
              canvas.pixels.begin() + static_cast<std::ptrdiff_t>(dstRow));
    std::fill(canvas.known.begin() + static_cast<std::ptrdiff_t>(dstRow),
              canvas.known.begin() + static_cast<std::ptrdiff_t>(dstRow + patch.width),
              static_cast<uint8_t>(1));
  }
  return true;
}

void absorbFrameIntoPanCanvas(
    std::shared_ptr<PanCanvas>& canvas,
    const RenderResult& frame,
    int palette,
    bool smoothColor) {
  if (frame.aborted || frame.width <= 0 || frame.height <= 0 || frame.pixels.empty()) {
    return;
  }

  if (canvas && canvas->viewWidth == frame.width && canvas->viewHeight == frame.height &&
      canvas->zoom == frame.zoom && canvas->maxIterations == frame.maxIterations &&
      canvas->samplesPerAxis == frame.samplesPerAxis && canvas->palette == palette &&
      canvas->smoothColor == smoothColor && mergeFrameIntoPanCanvas(*canvas, frame)) {
    canvas->stage = std::max(canvas->stage, frame.stage);
    return;
  }

  canvas = makePanCanvasFromFrame(frame, palette, smoothColor);
}

bool panCanvasMatchesCurrentView(
    const PanCanvas& canvas,
    const AppState& state,
    int clientWidth,
    int clientHeight) {
  if (!panCanvasReady(canvas) || canvas.zoom != state.zoom ||
      canvas.palette != state.palette || canvas.smoothColor != state.smoothColor) {
    return false;
  }

  const double canvasAspect = static_cast<double>(canvas.viewWidth) / static_cast<double>(std::max(1, canvas.viewHeight));
  const double currentAspect = static_cast<double>(std::max(1, clientWidth)) /
                               static_cast<double>(std::max(1, clientHeight));
  return std::abs(canvasAspect - currentAspect) < 0.001;
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
      std::lock_guard<std::mutex> lock(resultMutex_);
      completedPatches_.clear();
    }
    {
      std::lock_guard<std::mutex> lock(requestMutex_);
      pendingRequest_ = request;
    }
    requestCv_.notify_one();
  }

  void cancel(uint64_t sequence) {
    latestSequence_.store(sequence, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(resultMutex_);
      completedPatches_.clear();
    }
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

  bool takePatches(std::vector<RenderPatch>& patches) {
    std::lock_guard<std::mutex> lock(resultMutex_);
    if (completedPatches_.empty()) {
      return false;
    }

    patches = std::move(completedPatches_);
    completedPatches_.clear();
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
      progressTotal_.store(renderTileCount(request), std::memory_order_relaxed);
      auto publishPatch = [this](RenderPatch&& patch) {
        std::lock_guard<std::mutex> lock(resultMutex_);
        completedPatches_.push_back(std::move(patch));
      };
      RenderResult result = renderMandelbrot(request, latestSequence_, progressDone_, publishPatch);
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
  std::vector<RenderPatch> completedPatches_;
};

bool createDeviceD3D(HWND hwnd) {
  DXGI_SWAP_CHAIN_DESC desc = {};
  desc.BufferCount = 2;
  desc.BufferDesc.Width = 0;
  desc.BufferDesc.Height = 0;
  desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  desc.BufferDesc.RefreshRate.Numerator = 60;
  desc.BufferDesc.RefreshRate.Denominator = 1;
  desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
  desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  desc.OutputWindow = hwnd;
  desc.SampleDesc.Count = 1;
  desc.SampleDesc.Quality = 0;
  desc.Windowed = TRUE;
  desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

  UINT createDeviceFlags = 0;
  D3D_FEATURE_LEVEL featureLevel;
  const D3D_FEATURE_LEVEL featureLevelArray[2] = {
      D3D_FEATURE_LEVEL_11_0,
      D3D_FEATURE_LEVEL_10_0,
  };

  HRESULT hr = D3D11CreateDeviceAndSwapChain(
      nullptr,
      D3D_DRIVER_TYPE_HARDWARE,
      nullptr,
      createDeviceFlags,
      featureLevelArray,
      2,
      D3D11_SDK_VERSION,
      &desc,
      &g_swapChain,
      &g_device,
      &featureLevel,
      &g_deviceContext);

  return SUCCEEDED(hr);
}

void cleanupRenderTarget() {
  if (g_mainRenderTargetView) {
    g_mainRenderTargetView->Release();
    g_mainRenderTargetView = nullptr;
  }
}

void createRenderTarget() {
  ID3D11Texture2D* backBuffer = nullptr;
  g_swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
  if (backBuffer) {
    g_device->CreateRenderTargetView(backBuffer, nullptr, &g_mainRenderTargetView);
    backBuffer->Release();
  }
}

bool ensureComputeResources() {
  if (g_mandelbrotComputeShader && g_computeParamsBuffer) {
    return true;
  }

  HRESULT hr = g_device->CreateComputeShader(
      kMandelbrotComputeShaderBytes,
      kMandelbrotComputeShaderSize,
      nullptr,
      &g_mandelbrotComputeShader);
  if (FAILED(hr)) {
    return false;
  }

  D3D11_BUFFER_DESC bufferDesc = {};
  bufferDesc.ByteWidth = sizeof(GpuParams);
  bufferDesc.Usage = D3D11_USAGE_DEFAULT;
  bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  if (FAILED(g_device->CreateBuffer(&bufferDesc, nullptr, &g_computeParamsBuffer))) {
    return false;
  }

  return true;
}

bool ensureGpuFrameTexture(int width, int height) {
  if (width <= 0 || height <= 0) {
    return false;
  }

  if (g_frameTexture && g_frameTextureUav && g_frameTextureView && g_frameTextureWidth == width && g_frameTextureHeight == height) {
    return true;
  }

  if (g_frameTextureUav) {
    g_frameTextureUav->Release();
    g_frameTextureUav = nullptr;
  }
  if (g_frameTextureView) {
    g_frameTextureView->Release();
    g_frameTextureView = nullptr;
  }
  if (g_frameTexture) {
    g_frameTexture->Release();
    g_frameTexture = nullptr;
  }

  D3D11_TEXTURE2D_DESC desc = {};
  desc.Width = static_cast<UINT>(width);
  desc.Height = static_cast<UINT>(height);
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

  if (FAILED(g_device->CreateTexture2D(&desc, nullptr, &g_frameTexture))) {
    return false;
  }

  D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
  srvDesc.Format = desc.Format;
  srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
  srvDesc.Texture2D.MipLevels = 1;
  if (FAILED(g_device->CreateShaderResourceView(g_frameTexture, &srvDesc, &g_frameTextureView))) {
    return false;
  }

  D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
  uavDesc.Format = desc.Format;
  uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
  if (FAILED(g_device->CreateUnorderedAccessView(g_frameTexture, &uavDesc, &g_frameTextureUav))) {
    return false;
  }

  g_frameTextureWidth = width;
  g_frameTextureHeight = height;
  return true;
}

bool renderGpuPreview(const RenderRequest& request) {
  if (!ensureComputeResources() || !ensureGpuFrameTexture(request.width, request.height)) {
    return false;
  }

  const float aspect = static_cast<float>(request.width) / static_cast<float>(request.height);
  const float halfHeight = static_cast<float>(static_cast<double>(static_cast<Real>(1.5L) / request.zoom));
  const float halfWidth = halfHeight * aspect;

  GpuParams params = {};
  params.centerX = static_cast<float>(static_cast<double>(request.centerX));
  params.centerY = static_cast<float>(static_cast<double>(request.centerY));
  params.halfWidth = halfWidth;
  params.halfHeight = halfHeight;
  params.width = request.width;
  params.height = request.height;
  params.maxIterations = request.maxIterations;
  params.palette = request.palette;

  g_deviceContext->UpdateSubresource(g_computeParamsBuffer, 0, nullptr, &params, 0, 0);
  g_deviceContext->CSSetShader(g_mandelbrotComputeShader, nullptr, 0);
  g_deviceContext->CSSetConstantBuffers(0, 1, &g_computeParamsBuffer);
  g_deviceContext->CSSetUnorderedAccessViews(0, 1, &g_frameTextureUav, nullptr);
  g_deviceContext->Dispatch((request.width + 15) / 16, (request.height + 15) / 16, 1);

  ID3D11UnorderedAccessView* nullUav = nullptr;
  ID3D11Buffer* nullBuffer = nullptr;
  g_deviceContext->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
  g_deviceContext->CSSetConstantBuffers(0, 1, &nullBuffer);
  g_deviceContext->CSSetShader(nullptr, nullptr, 0);
  return true;
}

bool uploadFrameTexture(const RenderResult& frame) {
  if (frame.width <= 0 || frame.height <= 0 || frame.pixels.empty()) {
    return false;
  }

  if (g_frameTextureView) {
    g_frameTextureView->Release();
    g_frameTextureView = nullptr;
  }
  if (g_frameTextureUav) {
    g_frameTextureUav->Release();
    g_frameTextureUav = nullptr;
  }
  if (g_frameTexture) {
    g_frameTexture->Release();
    g_frameTexture = nullptr;
  }
  g_frameTextureWidth = 0;
  g_frameTextureHeight = 0;

  D3D11_TEXTURE2D_DESC desc = {};
  desc.Width = static_cast<UINT>(frame.width);
  desc.Height = static_cast<UINT>(frame.height);
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

  D3D11_SUBRESOURCE_DATA data = {};
  data.pSysMem = frame.pixels.data();
  data.SysMemPitch = static_cast<UINT>(frame.width * sizeof(uint32_t));

  if (FAILED(g_device->CreateTexture2D(&desc, &data, &g_frameTexture))) {
    return false;
  }

  D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc = {};
  viewDesc.Format = desc.Format;
  viewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
  viewDesc.Texture2D.MipLevels = 1;

  if (FAILED(g_device->CreateShaderResourceView(g_frameTexture, &viewDesc, &g_frameTextureView))) {
    if (g_frameTexture) {
      g_frameTexture->Release();
      g_frameTexture = nullptr;
    }
    return false;
  }

  g_frameTextureWidth = frame.width;
  g_frameTextureHeight = frame.height;
  return true;
}

bool updateFrameTexturePatch(const RenderPatch& patch) {
  if (!g_frameTexture || patch.width <= 0 || patch.height <= 0 ||
      patch.frameWidth != g_frameTextureWidth || patch.frameHeight != g_frameTextureHeight ||
      patch.x < 0 || patch.y < 0 ||
      patch.x + patch.width > g_frameTextureWidth ||
      patch.y + patch.height > g_frameTextureHeight ||
      patch.pixels.size() != static_cast<size_t>(patch.width) * static_cast<size_t>(patch.height)) {
    return false;
  }

  D3D11_BOX box = {};
  box.left = static_cast<UINT>(patch.x);
  box.top = static_cast<UINT>(patch.y);
  box.front = 0;
  box.right = static_cast<UINT>(patch.x + patch.width);
  box.bottom = static_cast<UINT>(patch.y + patch.height);
  box.back = 1;
  g_deviceContext->UpdateSubresource(
      g_frameTexture,
      0,
      &box,
      patch.pixels.data(),
      static_cast<UINT>(patch.width * sizeof(uint32_t)),
      0);
  return true;
}

void cleanupDeviceD3D() {
  if (g_frameTextureView) {
    g_frameTextureView->Release();
    g_frameTextureView = nullptr;
  }
  if (g_frameTextureUav) {
    g_frameTextureUav->Release();
    g_frameTextureUav = nullptr;
  }
  if (g_frameTexture) {
    g_frameTexture->Release();
    g_frameTexture = nullptr;
  }
  if (g_computeParamsBuffer) {
    g_computeParamsBuffer->Release();
    g_computeParamsBuffer = nullptr;
  }
  if (g_mandelbrotComputeShader) {
    g_mandelbrotComputeShader->Release();
    g_mandelbrotComputeShader = nullptr;
  }
  cleanupRenderTarget();
  if (g_swapChain) {
    g_swapChain->Release();
    g_swapChain = nullptr;
  }
  if (g_deviceContext) {
    g_deviceContext->Release();
    g_deviceContext = nullptr;
  }
  if (g_device) {
    g_device->Release();
    g_device = nullptr;
  }
}

Real screenToWorldX(float mouseX, int clientWidth, int clientHeight, const AppState& state) {
  const Real aspect = static_cast<Real>(clientWidth) / static_cast<Real>(std::max(1, clientHeight));
  const Real halfWidth = (1.5L / state.zoom) * aspect;
  const Real nx = (static_cast<Real>(mouseX) / static_cast<Real>(std::max(1, clientWidth))) * 2.0L - 1.0L;
  return state.centerX + nx * halfWidth;
}

Real screenToWorldY(float mouseY, int clientHeight, const AppState& state) {
  const Real halfHeight = 1.5L / state.zoom;
  const Real ny = 1.0L - (static_cast<Real>(mouseY) / static_cast<Real>(std::max(1, clientHeight))) * 2.0L;
  return state.centerY + ny * halfHeight;
}

void rememberTextureFrame(
    AppState& state,
    int width,
    int height,
    Real centerX,
    Real centerY,
    Real zoom,
    bool valid) {
  state.textureFrameValid = valid && width > 0 && height > 0 && zoom > 0.0L;
  state.textureFrameWidth = width;
  state.textureFrameHeight = height;
  state.textureFrameCenterX = centerX;
  state.textureFrameCenterY = centerY;
  state.textureFrameZoom = zoom;
}

void rememberTextureFrame(AppState& state, const RenderResult& frame, bool valid = true) {
  rememberTextureFrame(state, frame.width, frame.height, frame.centerX, frame.centerY, frame.zoom, valid);
}

bool screenZoomPreviewActive(const AppState& state, double nowSeconds, bool panMode) {
  return !panMode && state.textureFrameValid && g_frameTextureView &&
         (nowSeconds - state.lastWheelZoomSeconds) < 0.22;
}

void markInteraction(AppState& state, double nowSeconds) {
  state.lastInteractionSeconds = nowSeconds;
  state.viewRevision += 1;
  state.displayedStage = -1;
  state.submittedStage = -1;
  state.dirty = true;
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
  state.lastDragSeconds = -1000.0;
  state.lastWheelZoomSeconds = -1000.0;
  state.displayedViewRevision = 0;
  state.submittedViewRevision = 0;
  markInteraction(state, nowSeconds);
}

int chooseIterations(const AppState& state, bool moving) {
  int iterations = state.maxIterations;
  if (state.autoIterations) {
    const Real safeZoom = std::max(state.zoom, static_cast<Real>(1.0L));
    const int suggested = 360 + static_cast<int>(static_cast<double>(realLog10(safeZoom)) * 260.0);
    iterations = std::clamp(suggested, 320, state.maxIterations);
  }

  if (moving) {
    iterations = std::min(iterations, state.previewIterationCap);
  }

  return std::clamp(iterations, 32, kMaxIterationLimit);
}

int maxProgressiveStage(bool moving) {
  return moving ? 0 : 3;
}

int nextProgressiveStage(const AppState& state, bool moving) {
  if (state.dirty || state.displayedViewRevision != state.viewRevision) {
    return 0;
  }

  const int maxStage = maxProgressiveStage(moving);
  if (state.displayedStage >= 0 && state.displayedStage < maxStage) {
    return state.displayedStage + 1;
  }

  return -1;
}

float stageScale(int stage) {
  if (stage <= 0) {
    return 1.0f / 32.0f;
  }
  if (stage == 1) {
    return 0.125f;
  }
  if (stage == 2) {
    return 0.50f;
  }
  return 1.0f;
}

int stageIterations(const AppState& state, bool moving, int stage) {
  const int fullIterations = chooseIterations(state, moving);
  if (stage <= 0) {
    return std::clamp(std::min(fullIterations, state.previewIterationCap), 64, kMaxIterationLimit);
  }
  if (stage == 1) {
    return std::clamp(std::min(fullIterations, std::max(state.previewIterationCap * 2, 1000)), 64, kMaxIterationLimit);
  }
  if (stage == 2) {
    return std::clamp(std::min(fullIterations, std::max(state.previewIterationCap * 4, 2000)), 64, kMaxIterationLimit);
  }
  return fullIterations;
}

void updateAutoZoom(AppState& state, double dt) {
  if (!state.autoZoom) {
    return;
  }

  const Real follow = 1.0L - realExp(static_cast<Real>(-dt * 5.0));
  state.centerX += (state.targetX - state.centerX) * follow;
  state.centerY += (state.targetY - state.centerY) * follow;
  state.zoom = std::min(state.zoom * realPow(static_cast<Real>(state.zoomSpeed), static_cast<Real>(dt)), maxZoom());
  state.viewRevision += 1;
  state.displayedStage = -1;
  state.submittedStage = -1;
  state.dirty = true;
}

Real currentPixelStep(const AppState& state, int height) {
  return (3.0L / state.zoom) / static_cast<Real>(std::max(1, height));
}

void drawCurrentBackground(const AppState& state, int clientWidth, int clientHeight) {
  ImDrawList* drawList = ImGui::GetBackgroundDrawList();
  drawList->AddRectFilled(
      ImVec2(0.0f, 0.0f),
      ImVec2(static_cast<float>(clientWidth), static_cast<float>(clientHeight)),
      IM_COL32(4, 5, 5, 255));

  if (!g_frameTextureView) {
    return;
  }

  if (state.textureFrameValid && state.textureFrameWidth > 0 && state.textureFrameHeight > 0 &&
      state.zoom > 0.0L && state.textureFrameZoom > 0.0L) {
    const Real currentAspect =
        static_cast<Real>(std::max(1, clientWidth)) / static_cast<Real>(std::max(1, clientHeight));
    const Real oldAspect =
        static_cast<Real>(state.textureFrameWidth) / static_cast<Real>(std::max(1, state.textureFrameHeight));
    const Real currentHalfHeight = 1.5L / state.zoom;
    const Real currentHalfWidth = currentHalfHeight * currentAspect;
    const Real oldHalfHeight = 1.5L / state.textureFrameZoom;
    const Real oldHalfWidth = oldHalfHeight * oldAspect;

    const Real oldLeft = state.textureFrameCenterX - oldHalfWidth;
    const Real oldRight = state.textureFrameCenterX + oldHalfWidth;
    const Real oldTop = state.textureFrameCenterY + oldHalfHeight;
    const Real oldBottom = state.textureFrameCenterY - oldHalfHeight;

    const auto toScreenX = [&](Real x) {
      const Real normalized = (x - state.centerX) / currentHalfWidth;
      return static_cast<float>(static_cast<double>((normalized + 1.0L) * 0.5L *
                                                    static_cast<Real>(clientWidth)));
    };
    const auto toScreenY = [&](Real y) {
      const Real normalized = (y - state.centerY) / currentHalfHeight;
      return static_cast<float>(static_cast<double>((1.0L - normalized) * 0.5L *
                                                    static_cast<Real>(clientHeight)));
    };

    drawList->AddImage(
        reinterpret_cast<ImTextureID>(g_frameTextureView),
        ImVec2(toScreenX(oldLeft), toScreenY(oldTop)),
        ImVec2(toScreenX(oldRight), toScreenY(oldBottom)));
    return;
  }

  drawList->AddImage(
      reinterpret_cast<ImTextureID>(g_frameTextureView),
      ImVec2(0.0f, 0.0f),
      ImVec2(static_cast<float>(clientWidth), static_cast<float>(clientHeight)));
}

Real coordinateUlp(const AppState& state) {
  const Real inf = __builtin_huge_valq();
  const Real xUlp = realAbs(realNextAfter(state.centerX, inf) - state.centerX);
  const Real yUlp = realAbs(realNextAfter(state.centerY, inf) - state.centerY);
  return std::max(xUlp, yUlp);
}

Real gpuCoordinateUlp(const AppState& state) {
  const float inf = std::numeric_limits<float>::infinity();
  const float centerX = static_cast<float>(static_cast<double>(state.centerX));
  const float centerY = static_cast<float>(static_cast<double>(state.centerY));
  const float xUlp = std::abs(std::nextafter(centerX, inf) - centerX);
  const float yUlp = std::abs(std::nextafter(centerY, inf) - centerY);
  return static_cast<Real>(std::max({xUlp, yUlp, std::numeric_limits<float>::denorm_min()}));
}

bool gpuPreviewPreciseEnough(const AppState& state, int previewHeight) {
  const Real pixelStep = currentPixelStep(state, previewHeight);
  return pixelStep > gpuCoordinateUlp(state) * 2.0L;
}

int recommendedIterationsForScreen(const FrameStats& stats, int width, int height, int samplesPerAxis, int fallback) {
  if (stats.estimatedIterationsPerSecond <= 0.0) {
    return std::clamp(fallback, 64, kMaxIterationLimit);
  }

  const double sampleScale = static_cast<double>(std::max(1, samplesPerAxis)) *
                             static_cast<double>(std::max(1, samplesPerAxis));
  const double screenWorkPixels =
      static_cast<double>(std::max(1, width)) * static_cast<double>(std::max(1, height)) * sampleScale;
  const double recommended = stats.estimatedIterationsPerSecond * kRecommendedFullFrameSeconds / screenWorkPixels;
  return std::clamp(static_cast<int>(std::llround(recommended)), 64, kMaxIterationLimit);
}

void handleInput(AppState& state, int clientWidth, int clientHeight, double nowSeconds) {
  ImGuiIO& io = ImGui::GetIO();
  const bool mouseInWindow =
      io.MousePos.x >= 0.0f && io.MousePos.y >= 0.0f &&
      io.MousePos.x < static_cast<float>(clientWidth) &&
      io.MousePos.y < static_cast<float>(clientHeight);
  const bool canvasInput = mouseInWindow && !io.WantCaptureMouse;

  if (canvasInput && io.MouseWheel != 0.0f) {
    const Real beforeX = screenToWorldX(io.MousePos.x, clientWidth, clientHeight, state);
    const Real beforeY = screenToWorldY(io.MousePos.y, clientHeight, state);
    const Real factor = realPow(1.18L, static_cast<Real>(io.MouseWheel));
    state.zoom = std::clamp(state.zoom * factor, static_cast<Real>(0.05L), maxZoom());
    const Real afterX = screenToWorldX(io.MousePos.x, clientWidth, clientHeight, state);
    const Real afterY = screenToWorldY(io.MousePos.y, clientHeight, state);
    state.centerX += beforeX - afterX;
    state.centerY += beforeY - afterY;
    state.targetX = state.centerX;
    state.targetY = state.centerY;
    state.lastWheelZoomSeconds = nowSeconds;
    markInteraction(state, nowSeconds);
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
    const float distance2 = dx * dx + dy * dy;
    if (distance2 > 16.0f) {
      state.dragging = true;
    }

    if (state.dragging) {
      const Real aspect = static_cast<Real>(clientWidth) / static_cast<Real>(std::max(1, clientHeight));
      const Real worldPerPixelX = (3.0L / state.zoom * aspect) / static_cast<Real>(std::max(1, clientWidth));
      const Real worldPerPixelY = (3.0L / state.zoom) / static_cast<Real>(std::max(1, clientHeight));
      state.centerX = state.dragCenterX - static_cast<Real>(dx) * worldPerPixelX;
      state.centerY = state.dragCenterY + static_cast<Real>(dy) * worldPerPixelY;
      state.targetX = state.centerX;
      state.targetY = state.centerY;
      state.autoZoom = false;
      state.lastDragSeconds = nowSeconds;
      markInteraction(state, nowSeconds);
    }
  }

  if (state.mouseDown && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    if (!state.dragging && canvasInput) {
      state.targetX = screenToWorldX(io.MousePos.x, clientWidth, clientHeight, state);
      state.targetY = screenToWorldY(io.MousePos.y, clientHeight, state);
      state.autoZoom = true;
      markInteraction(state, nowSeconds);
    } else if (state.dragging) {
      state.lastDragSeconds = nowSeconds;
      markInteraction(state, nowSeconds);
    }
    state.mouseDown = false;
    state.dragging = false;
  }

  if (!io.WantCaptureKeyboard) {
    if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
      state.autoZoom = !state.autoZoom;
      markInteraction(state, nowSeconds);
    }
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
      resetView(state, nowSeconds);
    }
  }
}

void drawControls(
    AppState& state,
    const FrameStats& stats,
    bool rendererBusy,
    float renderProgress,
    double nowSeconds,
    int clientWidth,
    int clientHeight) {
  ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(390.0f, 0.0f), ImGuiCond_FirstUseEver);
  ImGui::Begin("曼德布罗集控制台");
  ImGui::TextUnformatted("公式：z(n+1) = z(n)^2 + C");
  ImGui::Separator();

  if (ImGui::Checkbox("自动放大（空格）", &state.autoZoom)) {
    markInteraction(state, nowSeconds);
  }
  if (ImGui::SliderFloat("放大速度", &state.zoomSpeed, 1.05f, 4.0f, "%.2f 倍/秒")) {
    markInteraction(state, nowSeconds);
  }

  const int screenRecommendedIterations =
      recommendedIterationsForScreen(stats, clientWidth, clientHeight, state.supersample, state.maxIterations);
  const int sliderIterationLimit =
      std::clamp(std::max(screenRecommendedIterations, state.maxIterations), 64, kMaxIterationLimit);

  if (ImGui::SliderInt("最大迭代次数", &state.maxIterations, 64, sliderIterationLimit)) {
    state.maxIterations = std::clamp(state.maxIterations, state.previewIterationCap, kMaxIterationLimit);
    markInteraction(state, nowSeconds);
  }
  if (ImGui::InputInt("手动输入最大迭代", &state.maxIterations, 100, 10000)) {
    state.maxIterations = std::clamp(state.maxIterations, state.previewIterationCap, kMaxIterationLimit);
    markInteraction(state, nowSeconds);
  }
  ImGui::Text(
      "屏幕建议最大迭代：%d（%.0f 秒/满屏）",
      screenRecommendedIterations,
      kRecommendedFullFrameSeconds);
  ImGui::SameLine();
  if (ImGui::Button("使用建议")) {
    state.maxIterations = std::clamp(screenRecommendedIterations, 64, kMaxIterationLimit);
    state.previewIterationCap = std::min(state.previewIterationCap, state.maxIterations);
    markInteraction(state, nowSeconds);
  }
  if (ImGui::SliderInt("移动时迭代上限", &state.previewIterationCap, 32, kMaxPreviewIterationLimit)) {
    state.previewIterationCap = std::min(state.previewIterationCap, state.maxIterations);
    markInteraction(state, nowSeconds);
  }
  if (ImGui::SliderInt("超采样轴数", &state.supersample, 1, 4, "%d")) {
    markInteraction(state, nowSeconds);
  }
  if (ImGui::Checkbox("级数逼近加速", &state.useSeriesApproximation)) {
    markInteraction(state, nowSeconds);
  }
  if (ImGui::SliderInt("级数最大跳步", &state.seriesMaxSkip, 16, kMaxSeriesSkipLimit)) {
    markInteraction(state, nowSeconds);
  }
  const int maxThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  state.cpuThreads = std::clamp(state.cpuThreads, 1, maxThreads);
  if (ImGui::SliderInt("CPU 工作线程", &state.cpuThreads, 1, maxThreads)) {
    markInteraction(state, nowSeconds);
  }
  ImGui::SameLine();
  if (ImGui::Button("拉满")) {
    state.cpuThreads = maxThreads;
    markInteraction(state, nowSeconds);
  }
  ImGui::Text("CPU 调度：动态小块抢任务（系统可用 %d 线程）", maxThreads);
  if (ImGui::Button("流畅预设")) {
    state.maxIterations = 5000;
    state.previewIterationCap = 500;
    state.supersample = 1;
    state.seriesMaxSkip = 2048;
    state.useSeriesApproximation = true;
    state.movingScale = 1.0f;
    state.stillScale = 1.0f;
    state.cpuThreads = maxThreads;
    markInteraction(state, nowSeconds);
  }
  ImGui::SameLine();
  if (ImGui::Button("画质预设")) {
    state.maxIterations = 12000;
    state.previewIterationCap = 900;
    state.supersample = 2;
    state.seriesMaxSkip = 4096;
    state.useSeriesApproximation = true;
    state.movingScale = 1.0f;
    state.stillScale = 1.00f;
    state.cpuThreads = maxThreads;
    markInteraction(state, nowSeconds);
  }
  ImGui::TextUnformatted("渐进渲染：滚轮先直接放大当前画面，停下后再 1/8、1/2、100% 补清晰");
  if (ImGui::Checkbox("自动调迭代次数", &state.autoIterations)) {
    markInteraction(state, nowSeconds);
  }
  if (ImGui::Checkbox("平滑着色", &state.smoothColor)) {
    markInteraction(state, nowSeconds);
  }
  if (ImGui::Checkbox("GPU 预览加速", &state.gpuPreview)) {
    markInteraction(state, nowSeconds);
  }

  const char* palettes[] = {"余弦色带", "热力色带", "印刷色带", "高对比霓虹"};
  if (ImGui::Combo("配色", &state.palette, palettes, IM_ARRAYSIZE(palettes))) {
    markInteraction(state, nowSeconds);
  }

  if (ImGui::Button("重置视图（R）")) {
    resetView(state, nowSeconds);
  }
  ImGui::SameLine();
  if (ImGui::Button("停止")) {
    state.autoZoom = false;
    markInteraction(state, nowSeconds);
  }

  ImGui::Separator();
  const std::string centerXText = formatReal(state.centerX, 28);
  const std::string centerYText = formatReal(state.centerY, 28);
  const std::string zoomText = formatReal(state.zoom, 18);
  ImGui::Text("中心 X：%s", centerXText.c_str());
  ImGui::Text("中心 Y：%s", centerYText.c_str());
  ImGui::Text("缩放倍数：%s", zoomText.c_str());
  ImGui::Text("数值精度：__float128，有效数字 %d 位，尾数 %d 位", kRealDigits10, kRealMantissaBits);
  const Real pixelStep = currentPixelStep(state, std::max(1, stats.displayedHeight));
  const Real ulp = coordinateUlp(state);
  const Real precisionRatio = pixelStep / ulp;
  const std::string pixelStepText = formatReal(pixelStep, 6);
  const std::string ulpText = formatReal(ulp, 6);
  const std::string precisionRatioText = formatReal(precisionRatio, 6);
  ImGui::Text("单像素跨度：%s", pixelStepText.c_str());
  ImGui::Text("坐标最小步长（ULP）：%s", ulpText.c_str());
  ImGui::Text("像素跨度 / ULP：%s %s",
              precisionRatioText.c_str(),
              precisionRatio < static_cast<Real>(16.0L) ? "（接近精度极限）" : "");
  ImGui::Text("预览来源：%s", stats.displayedGpuPreview ? "GPU 快速预览" : "CPU 高精度");
  if (stats.displayedPanReuse) {
    ImGui::TextUnformatted("拖动画布：内存画布即时显示，未知区域黑色，后台只补缺失小块");
  }
  if (state.gpuPreview && !stats.gpuPreviewPrecisionOk) {
    ImGui::TextUnformatted("GPU float 预览精度不足，已自动改用 CPU 预览");
  }
  ImGui::Text("当前帧：%dx%d，迭代 %d，抗锯齿 %dx%d",
              stats.displayedWidth,
              stats.displayedHeight,
              stats.displayedIterations,
              stats.displayedSamplesPerAxis,
              stats.displayedSamplesPerAxis);
  ImGui::Text("级数逼近：%s，当前跳过 %d 次/像素",
              state.useSeriesApproximation ? "开启" : "关闭",
              stats.displayedSeriesSkipIterations);
  ImGui::Text("CPU 线程：%d / %d", state.cpuThreads, maxThreads);
  ImGui::Text("CPU 计算耗时：%.2f 毫秒", stats.renderMilliseconds);
  const char* stageName = "未显示";
  if (stats.displayedStage <= 0) {
    stageName = "1/32 缩略预览";
  } else if (stats.displayedStage == 1) {
    stageName = "1/8 粗预览";
  } else if (stats.displayedStage == 2) {
    stageName = "1/2 补帧";
  } else {
    stageName = "全分辨率";
  }
  ImGui::Text("清晰阶段：%s", stageName);
  ImGui::Text("渲染状态：%s，帧序号 %llu", rendererBusy ? "正在计算" : "空闲",
              static_cast<unsigned long long>(stats.displayedSequence));
  char progressText[64] = {};
  std::snprintf(progressText, sizeof(progressText), "当前帧进度 %.1f%%", renderProgress * 100.0f);
  ImGui::ProgressBar(renderProgress, ImVec2(-1.0f, 0.0f), progressText);
  ImGui::Separator();
  ImGui::TextUnformatted("左键点击：选择放大目标");
  ImGui::TextUnformatted("左键拖动：平移视图");
  ImGui::TextUnformatted("滚轮：手动缩放");
  ImGui::TextUnformatted("Esc：退出");
  ImGui::End();
}

void submitFrameIfNeeded(
    CpuRenderer& renderer,
    AppState& state,
    const std::shared_ptr<const PanCanvas>& panCanvasSeed,
    int clientWidth,
    int clientHeight,
    bool moving,
    bool panMode,
    bool urgent,
    double nowSeconds,
    uint64_t& sequence) {
  if (clientWidth <= 0 || clientHeight <= 0) {
    return;
  }

  const int stage = nextProgressiveStage(state, moving);
  if (stage < 0 && !state.autoZoom) {
    return;
  }

  if (renderer.busy()) {
    const bool newView = state.submittedViewRevision != state.viewRevision;
    const bool thumbnailCanPreempt = stage == 0 && newView;
    const double minSubmitGap = urgent ? 0.0 : (moving ? 0.12 : 0.0);
    const bool canInterrupt = thumbnailCanPreempt && (nowSeconds - state.lastRenderSubmitSeconds) >= minSubmitGap;
    if (!canInterrupt) {
      return;
    }
  }

  if (stage < 0) {
    return;
  }

  if (state.submittedViewRevision == state.viewRevision && state.submittedStage == stage) {
    return;
  }

  const float scale = stageScale(stage);
  RenderRequest request;
  const bool usePanCanvas = panCanvasSeed && stage == 0 && moving;
  const bool dragWithoutCanvas = panMode && !panCanvasSeed;
  if (usePanCanvas) {
    request.width = panCanvasSeed->viewWidth;
    request.height = panCanvasSeed->viewHeight;
    request.maxIterations = panCanvasSeed->maxIterations;
    request.palette = panCanvasSeed->palette;
    request.smoothColor = panCanvasSeed->smoothColor;
    request.samplesPerAxis = panCanvasSeed->samplesPerAxis;
    request.panCanvas = panCanvasSeed;
  } else {
    const float renderScale = dragWithoutCanvas ? 1.0f : scale;
    request.width = std::max(80, static_cast<int>(static_cast<float>(clientWidth) * renderScale));
    request.height = std::max(60, static_cast<int>(static_cast<float>(clientHeight) * renderScale));
    request.maxIterations = stageIterations(state, moving, stage);
    request.palette = state.palette;
    request.smoothColor = state.smoothColor;
    request.samplesPerAxis = (stage >= 3 && !moving) ? state.supersample : 1;
  }
  request.useSeriesApproximation = state.useSeriesApproximation;
  request.workerCount = state.cpuThreads;
  request.stage = stage;
  request.seriesMaxSkip = state.seriesMaxSkip;
  request.publishPatches = panMode;
  request.viewRevision = state.viewRevision;
  request.centerX = state.centerX;
  request.centerY = state.centerY;
  request.zoom = state.zoom;
  request.sequence = ++sequence;
  renderer.submit(request);
  state.submittedViewRevision = state.viewRevision;
  state.submittedStage = stage;
  state.lastRenderSubmitSeconds = nowSeconds;
  state.dirty = false;
}

LRESULT WINAPI wndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam)) {
    return true;
  }

  switch (msg) {
    case WM_SIZE:
      if (wParam != SIZE_MINIMIZED) {
        g_resizeWidth = static_cast<UINT>(LOWORD(lParam));
        g_resizeHeight = static_cast<UINT>(HIWORD(lParam));
      }
      return 0;
    case WM_SYSCOMMAND:
      if ((wParam & 0xfff0) == SC_KEYMENU) {
        return 0;
      }
      break;
    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
    default:
      break;
  }
  return DefWindowProcW(hwnd, msg, wParam, lParam);
}

}  // namespace

int APIENTRY WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int) {
  WNDCLASSEXW wc = {};
  wc.cbSize = sizeof(wc);
  wc.style = CS_CLASSDC;
  wc.lpfnWndProc = wndProc;
  wc.hInstance = instance;
  wc.lpszClassName = L"MandelbrotImguiWindow";
  RegisterClassExW(&wc);

  const int screenWidth = GetSystemMetrics(SM_CXSCREEN);
  const int screenHeight = GetSystemMetrics(SM_CYSCREEN);

  HWND hwnd = CreateWindowW(
      wc.lpszClassName,
      L"曼德布罗集 CPU 实时放大",
      WS_OVERLAPPEDWINDOW,
      0,
      0,
      screenWidth,
      screenHeight,
      nullptr,
      nullptr,
      wc.hInstance,
      nullptr);

  if (!createDeviceD3D(hwnd)) {
    cleanupDeviceD3D();
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 1;
  }

  ShowWindow(hwnd, SW_MAXIMIZE);
  UpdateWindow(hwnd);
  createRenderTarget();

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  ImFontConfig fontConfig;
  fontConfig.OversampleH = 2;
  fontConfig.OversampleV = 2;
  fontConfig.PixelSnapH = true;
  if (!io.Fonts->AddFontFromFileTTF(
          "C:\\Windows\\Fonts\\msyh.ttc",
          18.0f,
          &fontConfig,
          io.Fonts->GetGlyphRangesChineseFull())) {
    io.Fonts->AddFontFromFileTTF(
        "C:\\Windows\\Fonts\\simhei.ttf",
        18.0f,
        &fontConfig,
        io.Fonts->GetGlyphRangesChineseFull());
  }
  ImGui::StyleColorsDark();
  ImGui_ImplWin32_Init(hwnd);
  ImGui_ImplDX11_Init(g_device, g_deviceContext);

  CpuRenderer cpuRenderer;
  AppState state;
  state.cpuThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  FrameStats stats;
  std::shared_ptr<PanCanvas> panCanvas;
  uint64_t sequence = 0;
  bool done = false;
  bool wasMoving = false;
  auto lastTick = Clock::now();
  const auto startTick = lastTick;

  while (!done) {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
      if (msg.message == WM_QUIT) {
        done = true;
      }
    }
    if (done) {
      break;
    }

    if (g_resizeWidth != 0 && g_resizeHeight != 0) {
      cleanupRenderTarget();
      g_swapChain->ResizeBuffers(0, g_resizeWidth, g_resizeHeight, DXGI_FORMAT_UNKNOWN, 0);
      g_resizeWidth = 0;
      g_resizeHeight = 0;
      createRenderTarget();
      state.viewRevision += 1;
      state.displayedStage = -1;
      state.submittedStage = -1;
      state.dirty = true;
      state.textureFrameValid = false;
      panCanvas.reset();
    }

    RECT clientRect = {};
    GetClientRect(hwnd, &clientRect);
    const int clientWidth = std::max(1L, clientRect.right - clientRect.left);
    const int clientHeight = std::max(1L, clientRect.bottom - clientRect.top);

    const auto now = Clock::now();
    const double dt = std::chrono::duration<double>(now - lastTick).count();
    const double nowSeconds = std::chrono::duration<double>(now - startTick).count();
    lastTick = now;

    updateAutoZoom(state, std::clamp(dt, 0.0, 0.1));

    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    handleInput(state, clientWidth, clientHeight, nowSeconds);

    const bool recentlyInteracted = (nowSeconds - state.lastInteractionSeconds) < 0.20;
    const bool panMode = state.dragging || (nowSeconds - state.lastDragSeconds) < 0.35;
    const bool moving = state.autoZoom || panMode || recentlyInteracted;
    const bool zoomPreviewMode = screenZoomPreviewActive(state, nowSeconds, panMode);
    if (zoomPreviewMode && state.displayedViewRevision != state.viewRevision) {
      cpuRenderer.cancel(++sequence);
      state.displayedViewRevision = state.viewRevision;
      state.displayedStage = 0;
      state.submittedViewRevision = state.viewRevision;
      state.submittedStage = 0;
      state.dirty = false;
    }
    if (wasMoving && !moving) {
      state.dirty = state.displayedViewRevision != state.viewRevision;
    }
    wasMoving = moving;

    if (panCanvas && !panCanvasMatchesCurrentView(*panCanvas, state, clientWidth, clientHeight)) {
      panCanvas.reset();
    }

    std::shared_ptr<const PanCanvas> panCanvasSeed;
    if (panMode && panCanvas && panCanvasMatchesCurrentView(*panCanvas, state, clientWidth, clientHeight)) {
      RenderResult canvasFrame =
          cropPanCanvasView(*panCanvas, state.centerX, state.centerY, state.viewRevision, ++sequence);
      if (uploadFrameTexture(canvasFrame)) {
        rememberTextureFrame(state, canvasFrame);
        stats.displayedWidth = canvasFrame.width;
        stats.displayedHeight = canvasFrame.height;
        stats.displayedIterations = canvasFrame.maxIterations;
        stats.displayedSamplesPerAxis = canvasFrame.samplesPerAxis;
        stats.displayedStage = canvasFrame.stage;
        stats.displayedSeriesSkipIterations = 0;
        stats.renderMilliseconds = 0.0;
        stats.displayedSequence = canvasFrame.sequence;
        stats.displayedGpuPreview = false;
        stats.displayedPanReuse = true;
      }
      // The CPU worker must read an immutable snapshot. The GUI thread may merge
      // finished patches into panCanvas while the worker is still rendering.
      panCanvasSeed = std::make_shared<PanCanvas>(*panCanvas);
    }

    const int previewStage = nextProgressiveStage(state, moving);
    const float previewScale = stageScale(previewStage);
    const int previewWidth = std::max(80, static_cast<int>(static_cast<float>(clientWidth) * previewScale));
    const int previewHeight = std::max(60, static_cast<int>(static_cast<float>(clientHeight) * previewScale));
    const bool gpuPrecisionOk = previewStage >= 0 && gpuPreviewPreciseEnough(state, previewHeight);
    stats.gpuPreviewPrecisionOk = previewStage < 0 || gpuPrecisionOk;
    if (panMode && !panCanvasSeed) {
      RenderResult blackFrame;
      blackFrame.width = clientWidth;
      blackFrame.height = clientHeight;
      blackFrame.maxIterations = 0;
      blackFrame.samplesPerAxis = 1;
      blackFrame.stage = 0;
      blackFrame.sequence = ++sequence;
      blackFrame.viewRevision = state.viewRevision;
      blackFrame.centerX = state.centerX;
      blackFrame.centerY = state.centerY;
      blackFrame.zoom = state.zoom;
      blackFrame.fromPanReuse = true;
      blackFrame.pixels.assign(
          static_cast<size_t>(blackFrame.width) * static_cast<size_t>(blackFrame.height),
          packBgra(4, 5, 5));
      if (uploadFrameTexture(blackFrame)) {
        rememberTextureFrame(state, blackFrame, false);
        stats.displayedWidth = blackFrame.width;
        stats.displayedHeight = blackFrame.height;
        stats.displayedIterations = 0;
        stats.displayedSamplesPerAxis = 1;
        stats.displayedStage = 0;
        stats.displayedSeriesSkipIterations = 0;
        stats.renderMilliseconds = 0.0;
        stats.displayedSequence = blackFrame.sequence;
        stats.displayedGpuPreview = false;
        stats.displayedPanReuse = true;
      }
    }

    if (!zoomPreviewMode && !panMode && !panCanvasSeed && state.gpuPreview && previewStage >= 0 &&
        previewStage < 3 && gpuPrecisionOk) {
      RenderRequest gpuRequest;
      gpuRequest.width = previewWidth;
      gpuRequest.height = previewHeight;
      gpuRequest.maxIterations = stageIterations(state, moving, previewStage);
      gpuRequest.palette = state.palette;
      gpuRequest.stage = previewStage;
      gpuRequest.viewRevision = state.viewRevision;
      gpuRequest.centerX = state.centerX;
      gpuRequest.centerY = state.centerY;
      gpuRequest.zoom = state.zoom;
      if (renderGpuPreview(gpuRequest)) {
        rememberTextureFrame(
            state,
            gpuRequest.width,
            gpuRequest.height,
            gpuRequest.centerX,
            gpuRequest.centerY,
            gpuRequest.zoom,
            true);
        stats.displayedWidth = gpuRequest.width;
        stats.displayedHeight = gpuRequest.height;
        stats.displayedIterations = gpuRequest.maxIterations;
        stats.displayedSamplesPerAxis = 1;
        stats.displayedStage = gpuRequest.stage;
        stats.displayedSeriesSkipIterations = 0;
        stats.renderMilliseconds = 0.0;
        stats.displayedSequence = ++sequence;
        stats.displayedGpuPreview = true;
        stats.displayedPanReuse = false;
        cpuRenderer.cancel(sequence);
        state.displayedViewRevision = gpuRequest.viewRevision;
        state.displayedStage = gpuRequest.stage;
        state.submittedViewRevision = 0;
        state.submittedStage = -1;
        state.dirty = false;
      } else {
        state.gpuPreview = false;
      }
    }

    RenderResult completedFrame;
    if (cpuRenderer.takeResult(completedFrame)) {
      if (completedFrame.viewRevision == state.viewRevision && uploadFrameTexture(completedFrame)) {
        rememberTextureFrame(state, completedFrame);
        stats.displayedWidth = completedFrame.width;
        stats.displayedHeight = completedFrame.height;
        stats.displayedIterations = completedFrame.maxIterations;
        stats.displayedSamplesPerAxis = completedFrame.samplesPerAxis;
        stats.displayedStage = completedFrame.stage;
        stats.displayedSeriesSkipIterations = completedFrame.seriesSkipIterations;
        stats.renderMilliseconds = completedFrame.milliseconds;
        stats.displayedSequence = completedFrame.sequence;
        stats.displayedGpuPreview = false;
        stats.displayedPanReuse = completedFrame.fromPanReuse;
        if (completedFrame.milliseconds > 0.0 && completedFrame.maxIterations > 0) {
          const double sampleScale = static_cast<double>(completedFrame.samplesPerAxis) *
                                     static_cast<double>(completedFrame.samplesPerAxis);
          const double testedIterations =
              static_cast<double>(completedFrame.width) * static_cast<double>(completedFrame.height) *
              sampleScale * static_cast<double>(completedFrame.maxIterations);
          stats.estimatedIterationsPerSecond = testedIterations / (completedFrame.milliseconds / 1000.0);
        }
        absorbFrameIntoPanCanvas(panCanvas, completedFrame, state.palette, state.smoothColor);
        state.displayedViewRevision = completedFrame.viewRevision;
        state.displayedStage = completedFrame.stage;
      }
    }

    std::vector<RenderPatch> completedPatches;
    if (cpuRenderer.takePatches(completedPatches)) {
      for (const RenderPatch& patch : completedPatches) {
        if (patch.viewRevision != state.viewRevision) {
          continue;
        }
        if (panCanvas) {
          mergePatchIntoPanCanvas(*panCanvas, patch);
        }
        if (updateFrameTexturePatch(patch)) {
          stats.displayedWidth = patch.frameWidth;
          stats.displayedHeight = patch.frameHeight;
          stats.displayedGpuPreview = false;
          stats.displayedPanReuse = true;
        }
      }
    }

    drawCurrentBackground(state, clientWidth, clientHeight);

    drawControls(state, stats, cpuRenderer.busy(), cpuRenderer.progress(), nowSeconds, clientWidth, clientHeight);
    submitFrameIfNeeded(
        cpuRenderer,
        state,
        panCanvasSeed,
        clientWidth,
        clientHeight,
        moving,
        panMode,
        recentlyInteracted || panMode,
        nowSeconds,
        sequence);

    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      done = true;
    }

    ImGui::Render();
    const float clearColor[4] = {0.015f, 0.018f, 0.018f, 1.0f};
    g_deviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
    g_deviceContext->ClearRenderTargetView(g_mainRenderTargetView, clearColor);
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    g_swapChain->Present(1, 0);
  }

  ImGui_ImplDX11_Shutdown();
  ImGui_ImplWin32_Shutdown();
  ImGui::DestroyContext();

  cleanupDeviceD3D();
  DestroyWindow(hwnd);
  UnregisterClassW(wc.lpszClassName, wc.hInstance);
  return 0;
}
