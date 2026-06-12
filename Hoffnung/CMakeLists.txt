cmake_minimum_required(VERSION 3.18)
project(gbdt-core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Find Dependencies ----
# LibTorch (PyTorch C++ API)
find_package(Torch REQUIRED)

# pybind11 for Python bindings
find_package(pybind11 REQUIRED)

# OpenMP for parallel histogram
find_package(OpenMP REQUIRED)

# ---- Sources ----
set(GBDT_SOURCES
    src/tree.cpp
    src/builder.cpp
    src/gbdt.cpp
    src/inference_engine.cpp
)

set(GBDT_HEADERS
    include/gbdt/config.h
    include/gbdt/tree.h
    include/gbdt/builder.h
    include/gbdt/gbdt.h
    include/gbdt/inference_engine.h
    include/gbdt/json_utils.h
)

# ---- Object Library (compile GBDT sources only once) ----
add_library(gbdt_objects OBJECT ${GBDT_SOURCES})
target_include_directories(gbdt_objects PRIVATE include)
target_link_libraries(gbdt_objects PRIVATE
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)
target_compile_features(gbdt_objects PUBLIC cxx_std_17)

# ---- Shared Library (C++ only, no Python) ----
add_library(gbdt_core SHARED $<TARGET_OBJECTS:gbdt_objects>)
target_include_directories(gbdt_core PUBLIC include)
target_link_libraries(gbdt_core PRIVATE
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)
target_compile_features(gbdt_core PUBLIC cxx_std_17)

# ---- Python Module (pybind11) ----
pybind11_add_module(gbdt_python src/python_bindings.cpp $<TARGET_OBJECTS:gbdt_objects>)
target_include_directories(gbdt_python PRIVATE include)
target_link_libraries(gbdt_python PRIVATE
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)
target_compile_features(gbdt_python PUBLIC cxx_std_17)

# ---- Tests ----
add_executable(test_tree tests/test_tree.cpp $<TARGET_OBJECTS:gbdt_objects>)
target_include_directories(test_tree PRIVATE include)
target_link_libraries(test_tree PRIVATE
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)
target_compile_features(test_tree PUBLIC cxx_std_17)

# Copy Python package to build dir
file(COPY python/ DESTINATION ${CMAKE_BINARY_DIR}/python)

# ---- Install ----
install(TARGETS gbdt_core gbdt_python
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES ${GBDT_HEADERS} DESTINATION include/gbdt)
install(DIRECTORY python/gbdt DESTINATION ${CMAKE_BINARY_DIR}/python)
