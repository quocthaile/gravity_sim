cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

if(DEFINED CMAKE_TOOLCHAIN_FILE)
    if(NOT DEFINED ENV{VCPKG_ROOT} OR "$ENV{VCPKG_ROOT}" STREQUAL "")
        message(FATAL_ERROR
            "VCPKG_ROOT is required when CMAKE_TOOLCHAIN_FILE is set. "
            "Set VCPKG_ROOT to the root of your vcpkg checkout."
        )
    endif()
    if(NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}")
        message(FATAL_ERROR
            "vcpkg toolchain file not found: ${CMAKE_TOOLCHAIN_FILE}. "
            "Check that VCPKG_ROOT points to a valid vcpkg checkout."
        )
    endif()
endif()

project(gravity_sim VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenMP QUIET)

set(GRAVITY_SIM_CORE_LIBS
    glfw
    GLEW::GLEW
    glm::glm
    OpenGL::GL
)

function(configure_gravity_target_warnings target)
    target_compile_features(${target} PUBLIC cxx_std_20)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
        target_compile_definitions(${target} PUBLIC
            WIN32_LEAN_AND_MEAN
            NOMINMAX
        )
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra)
        if(WIN32)
            target_compile_definitions(${target} PUBLIC
                WIN32_LEAN_AND_MEAN
                NOMINMAX
            )
        endif()
    endif()
endfunction()

# Single serialized asset-staging target so parallel Ninja links never race copying shaders/DLLs
add_custom_target(gravity_sim_stage_assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMENT "Staging shader and preset assets to build directory"
)

function(copy_gravity_runtime_assets target)
    add_dependencies(${target} gravity_sim_stage_assets)
    if(WIN32 AND "${target}" STREQUAL "gravity_sim_3Dgrid")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target}>
                $<TARGET_FILE_DIR:${target}>
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders
                $<TARGET_FILE_DIR:${target}>/shaders
            COMMAND_EXPAND_LISTS
        )
    endif()
endfunction()

# ============================================================================
# Shared Engine Static Library (all unified runtime, modules F1..F6 + Diag, UI)
# ============================================================================
add_library(gravity_sim_engine STATIC
    src/gpu_memory_manager.cpp
    src/gravity_sim_3Dgrid_function.cpp
    src/render/asset_paths.cpp
    src/render/resource_census.cpp
    src/render/shader_cache.cpp
    src/render/fullscreen_presenter.cpp
    src/render/shared_grid.cpp
    src/telemetry/telemetry_service.cpp
    src/benchmark/benchmark_runner.cpp
    src/app/camera_controllers.cpp
    src/app/simulation_manager.cpp
    src/app/application.cpp
    src/models/black_hole/black_hole_core.cpp
    src/models/black_hole/black_hole_gpu_module.cpp
    src/models/black_hole/black_hole_cpu_module.cpp
    src/models/gravity/gravity_gpu_module.cpp
    src/models/gravity_baseline/gravity_baseline_module.cpp
    src/models/light_trajectories/light_trajectories_module.cpp
    src/models/ray_tracing/flat_ray_tracing_module.cpp
    src/models/diagnostic/diagnostic_triangle_module.cpp
)

target_include_directories(gravity_sim_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gravity_sim_engine PUBLIC
    ${GRAVITY_SIM_CORE_LIBS}
    imgui::imgui
)

if(WIN32)
    target_link_libraries(gravity_sim_engine PUBLIC
        psapi
        dxgi
        uuid
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(gravity_sim_engine PUBLIC OpenMP::OpenMP_CXX)
endif()

configure_gravity_target_warnings(gravity_sim_engine)

# ============================================================================
# Primary Unified Application Executable
# ============================================================================
add_executable(gravity_sim_3Dgrid
    src/gravity_sim_3Dgrid.cpp
)
target_link_libraries(gravity_sim_3Dgrid PRIVATE gravity_sim_engine)
configure_gravity_target_warnings(gravity_sim_3Dgrid)
copy_gravity_runtime_assets(gravity_sim_3Dgrid)

# ============================================================================
# Standalone Historical Reference Executable (untouched baseline)
# ============================================================================
add_executable(gravity_sim_3Dgrid_baseline
    src/gravity_sim_3Dgrid_baseline.cpp
)
target_include_directories(gravity_sim_3Dgrid_baseline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(gravity_sim_3Dgrid_baseline PRIVATE ${GRAVITY_SIM_CORE_LIBS})
configure_gravity_target_warnings(gravity_sim_3Dgrid_baseline)
copy_gravity_runtime_assets(gravity_sim_3Dgrid_baseline)

# ============================================================================
# Automated CTest Acceptance Suite (T01 - T16)
# ============================================================================
include(CTest)
if(BUILD_TESTING)
    function(add_gravity_test_binary target source_file)
        add_executable(${target} ${source_file})
        target_link_libraries(${target} PRIVATE gravity_sim_engine)
        configure_gravity_target_warnings(${target})
        copy_gravity_runtime_assets(${target})
    endfunction()

    add_gravity_test_binary(test_phase2_and_grid_parity
        tests/test_phase2_and_grid_parity.cpp
    )
    add_gravity_test_binary(test_lifecycle_and_transitions
        tests/test_lifecycle_and_transitions.cpp
    )
    add_gravity_test_binary(test_geodesic_numerical_and_benchmark
        tests/test_geodesic_numerical_and_benchmark.cpp
    )

    add_test(NAME T01_GravityGpuParity
        COMMAND test_phase2_and_grid_parity --case T01)
    add_test(NAME T02_SharedGridSourcesAndZeroObjectRoles
        COMMAND test_phase2_and_grid_parity --case T02)
    add_test(NAME T03_SingleContextAndGridHandleInvariance
        COMMAND test_lifecycle_and_transitions --case T03)
    add_test(NAME T04_AllModePairsAndPresetResets
        COMMAND test_lifecycle_and_transitions --case T04)
    add_test(NAME T05_KeyRepeatIgnoreAndCoalescing
        COMMAND test_lifecycle_and_transitions --case T05)
    add_test(NAME T06_ExitDuringInitializingPausedAndExport
        COMMAND test_lifecycle_and_transitions --case T06)
    add_test(NAME T07_CpuWorkerCancellationAndStaleDrop
        COMMAND test_lifecycle_and_transitions --case T07)
    add_test(NAME T08_ResizeFullscreenMinimizeZeroByZeroGuard
        COMMAND test_lifecycle_and_transitions --case T08)
    add_test(NAME T09_FiftyCycleStressResourceCensusBounded
        COMMAND test_lifecycle_and_transitions --case T09)
    add_test(NAME T10_InitFailurePartialCleanupAndErrorRecovery
        COMMAND test_lifecycle_and_transitions --case T10)
    add_test(NAME T11_PausedGridDirtyAndSharedBlackHoleSync
        COMMAND test_phase2_and_grid_parity --case T11)
    add_test(NAME T12_BenchmarkConfigLockAndMutationAbort
        COMMAND test_geodesic_numerical_and_benchmark --case T12)
    add_test(NAME T13_GeodesicConvergenceInvariantsAndCpuGpuParity
        COMMAND test_geodesic_numerical_and_benchmark --case T13)
    add_test(NAME T14_F2BoundedTrailsSingleDrawZeroTrailCopy
        COMMAND test_lifecycle_and_transitions --case T14)
    add_test(NAME T15_ExternalWorkingDirectoryAssetResolution
        COMMAND test_geodesic_numerical_and_benchmark --case T15)
    add_test(NAME T16_HistoricalBaselineCpuPhysicsParity
        COMMAND test_geodesic_numerical_and_benchmark --case T16)
endif()

if(CMAKE_GENERATOR MATCHES "^Visual Studio")
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT gravity_sim_3Dgrid)
endif()
