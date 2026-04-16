cbuffer Params : register(b0) {
  float centerX;
  float centerY;
  float halfWidth;
  float halfHeight;
  int width;
  int height;
  int maxIterations;
  int palette;
};

RWTexture2D<float4> outputTexture : register(u0);

float3 paletteColor(float t) {
  t = saturate(t);
  if (palette == 1) {
    return float3(0.15 + 0.85 * pow(t, 0.38), 0.08 + 0.74 * sin(t * 1.5707963), 0.37 + 0.47 * (1.0 - t));
  }
  if (palette == 2) {
    float band = 0.5 + 0.5 * cos(6.2831853 * (t * 12.0));
    float base = pow(t, 0.22);
    return float3(base, base * (0.72 + 0.28 * band), base * 0.78);
  }
  if (palette == 3) {
    float u = t * 18.0;
    float band = floor(u);
    float local = frac(u);
    float hard = local < 0.58 ? 1.0 : 0.05;
    float m = fmod(band, 6.0);
    float3 c = float3(1.0, 0.96, 0.08);
    if (m >= 1.0 && m < 2.0) {
      c = float3(0.0, 0.92, 1.0);
    } else if (m >= 2.0 && m < 3.0) {
      c = float3(1.0, 0.0, 0.86);
    } else if (m >= 3.0 && m < 4.0) {
      c = float3(1.0, 1.0, 1.0);
    } else if (m >= 4.0 && m < 5.0) {
      c = float3(0.16, 1.0, 0.0);
    } else if (m >= 5.0) {
      c = float3(1.0, 0.31, 0.0);
    }
    return c * hard;
  }
  return 0.5 + 0.5 * cos(6.2831853 * (float3(t, t, t) + float3(0.04, 0.34, 0.67)));
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= (uint)width || id.y >= (uint)height) {
    return;
  }

  float nx = ((float(id.x) + 0.5f) / float(width)) * 2.0f - 1.0f;
  float ny = 1.0f - ((float(id.y) + 0.5f) / float(height)) * 2.0f;
  float cx = centerX + nx * halfWidth;
  float cy = centerY + ny * halfHeight;

  float x = 0.0f;
  float y = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  int iter = 0;

  [loop]
  for (; iter < maxIterations; ++iter) {
    if (x2 + y2 > 4.0f) {
      break;
    }
    y = 2.0f * x * y + cy;
    x = x2 - y2 + cx;
    x2 = x * x;
    y2 = y * y;
  }

  if (iter >= maxIterations) {
    outputTexture[id.xy] = float4(0.015, 0.018, 0.018, 1.0);
    return;
  }

  float t = sqrt(float(iter) / float(maxIterations));
  outputTexture[id.xy] = float4(paletteColor(t), 1.0);
}
