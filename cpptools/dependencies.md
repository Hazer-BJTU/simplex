# Dependencies
This document lists all third-party libraries and their dependencies required to build and run this C++ project. 

## Notes
The versions listed below are **development-phase dependencies** and are subject to change in subsequent release versions of the project. Updated dependency versions will be announced in subsequent release notes.

## Third-Party Libraries
| Library Name | Version | Official Website | GitHub Repository | License |
|--------------|---------|------------------|-------------------|---------|
| boost | 1.74.0 | https://www.boost.org/ | https://github.com/boostorg/boost | Boost Software License 1.0 |
| tree-sitter | 0.26.5 | https://tree-sitter.github.io/tree-sitter/ | https://github.com/tree-sitter/tree-sitter | MIT License |
| tree-sitter-python | 0.25.0 | https://tree-sitter.github.io/tree-sitter/ | https://github.com/tree-sitter/tree-sitter-python | MIT License |
| nlohmann/json | 3.12.0 | https://json.nlohmann.me/ | https://github.com/nlohmann/json | MIT License |

## Build Guidelines

### Prerequisites

Before building this project, ensure you have the following installed:

- **CMake** 3.20 or higher
- **C++20 compliant compiler** (GCC 10+, Clang 10+, or MSVC 19.28+)
- **Boost** 1.74.0 or higher with the following modules:
  - `filesystem`
  - `system`
  - `program_options`

> **Note:** Boost must be installed system-wide. The project uses `find_package(Boost REQUIRED)` to locate it.

### Pre-bundled Dependencies

The following dependencies are bundled in the `third_party` directory and do not require additional installation:

| Dependency | Location | Notes |
|------------|----------|-------|
| tree-sitter | `third_party/libs/libtree-sitter.a` | Pre-compiled static library |
| tree-sitter-python | `third_party/libs/libtree-sitter-python.a` | Pre-compiled static library |
| nlohmann/json | `third_party/include/nlohmann/` | Header-only library |

### Build Commands

Navigate to the `cpptools` directory and run the following commands:

```bash
# Create and enter build directory
mkdir -p build && cd build

# Configure the project
cmake ..

# Build the project
cmake --build .
```

After a successful build, the outputs will be located in:
- **Executable**: `bin/simplex_tool_server`
- **Static Library**: `libs/lib_simplex_lang.a`
- **Unit Tests**: `bin/tests/` (if BUILD_UNITEST is enabled)

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_RELEASE` | OFF | Enable release build with `-O3` optimization and LTO (Link Time Optimization) |
| `BUILD_UNITEST` | ON | Build unit test executables |

To enable release build:
```bash
cmake -DBUILD_RELEASE=ON ..
```

To disable unit tests:
```bash
cmake -DBUILD_UNITEST=OFF ..
```

### Platform Support

This project is designed for Unix-like systems (Linux, macOS). Windows support may require additional configuration.
