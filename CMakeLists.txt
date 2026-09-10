cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(gravity_sim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

set(GRAVITY_SIM_LIBS
    glfw
    GLEW::GLEW
    glm::glm
    OpenGL::GL
)

function(add_gravity_executable target)
    add_executable(${target} ${ARGN})

    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(${target} PRIVATE ${GRAVITY_SIM_LIBS})

    target_compile_features(${target} PRIVATE cxx_std_23)

    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
        target_compile_definitions(${target} PRIVATE
            WIN32_LEAN_AND_MEAN
            NOMINMAX
        )
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()

    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target}>
                $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS
        )
    endif()
endfunction()

add_gravity_executable(3d_test src/3D_test.cpp)
add_gravity_executable(gravity_sim src/gravity_sim.cpp)
add_gravity_executable(gravity_sim_3Dgrid
    src/gravity_sim_3Dgrid.cpp
    src/gravity_sim_3Dgrid_function.cpp
    shaders/vertex_shader.glsl
    shaders/fragment_shader.glsl
)

add_custom_command(TARGET gravity_sim_3Dgrid POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:gravity_sim_3Dgrid>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vertex_shader.glsl
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fragment_shader.glsl
        $<TARGET_FILE_DIR:gravity_sim_3Dgrid>/shaders
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT gravity_sim_3Dgrid)
