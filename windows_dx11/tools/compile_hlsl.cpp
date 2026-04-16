#include <windows.h>
#include <d3dcompiler.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<char> readFile(const char* path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open input shader");
  }
  return std::vector<char>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void writeHeader(const char* path, const void* data, size_t size) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open output header");
  }

  const auto* bytes = static_cast<const uint8_t*>(data);
  file << "#pragma once\n";
  file << "#include <cstddef>\n\n";
  file << "static constexpr unsigned char kMandelbrotComputeShaderBytes[] = {\n";
  for (size_t i = 0; i < size; ++i) {
    if ((i % 12) == 0) {
      file << "  ";
    }
    file << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(bytes[i]);
    if (i + 1 != size) {
      file << ", ";
    }
    if ((i % 12) == 11 || i + 1 == size) {
      file << "\n";
    }
  }
  file << "};\n";
  file << "static constexpr std::size_t kMandelbrotComputeShaderSize = sizeof(kMandelbrotComputeShaderBytes);\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: compile_hlsl <input.hlsl> <output.h>\n";
    return 2;
  }

  try {
    const std::vector<char> source = readFile(argv[1]);
    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    const HRESULT hr = D3DCompile(
        source.data(),
        source.size(),
        argv[1],
        nullptr,
        nullptr,
        "main",
        "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &shaderBlob,
        &errorBlob);

    if (FAILED(hr)) {
      if (errorBlob) {
        std::cerr.write(
            static_cast<const char*>(errorBlob->GetBufferPointer()),
            static_cast<std::streamsize>(errorBlob->GetBufferSize()));
        errorBlob->Release();
      }
      return 1;
    }
    if (errorBlob) {
      errorBlob->Release();
    }

    writeHeader(argv[2], shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize());
    shaderBlob->Release();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
