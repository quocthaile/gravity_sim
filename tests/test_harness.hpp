#pragma once

#include "app/application.hpp"
#include <GL/glew.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace grav_test
{

inline void FailAssertion(const char *expr, const char *file, int line, const std::string &msg = {})
{
    std::ostringstream oss;
    oss << "[TEST FAILURE] " << file << ":" << line << " — Assertion failed: (" << expr << ")";
    if (!msg.empty())
    {
        oss << " | " << msg;
    }
    throw std::runtime_error(oss.str());
}

inline void AssertNoGlError(const char *contextLabel, const char *file, int line)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        std::ostringstream oss;
        oss << "OpenGL error 0x" << std::hex << err << std::dec << " after " << contextLabel;
        FailAssertion("glGetError() == GL_NO_ERROR", file, line, oss.str());
    }
}

inline bool StepUntilSettled(grav_app::Application &app, double maxWallSeconds = 5.0,
                             double fixedDt = 1.0 / 60.0)
{
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 10000; ++i)
    {
        app.StepFrame(fixedDt);
        const auto st = app.SimManager().State();
        if (!app.SimManager().IsTransitioning() &&
            (st == grav_app::LifecycleState::Running || st == grav_app::LifecycleState::Error))
        {
            return true;
        }
        if (st == grav_app::LifecycleState::Quiescing)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (elapsed > maxWallSeconds)
        {
            break;
        }
    }
    return false;
}

inline std::string ParseCaseArg(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--case" && i + 1 < argc)
        {
            return argv[i + 1];
        }
    }
    return "ALL";
}

} // namespace grav_test

#define GRAV_ASSERT(expr)                                                                          \
    do                                                                                             \
    {                                                                                              \
        if (!(expr))                                                                               \
        {                                                                                          \
            ::grav_test::FailAssertion(#expr, __FILE__, __LINE__);                                 \
        }                                                                                          \
    } while (false)

#define GRAV_ASSERT_MSG(expr, msg)                                                                 \
    do                                                                                             \
    {                                                                                              \
        if (!(expr))                                                                               \
        {                                                                                          \
            std::ostringstream _grav_oss;                                                          \
            _grav_oss << msg;                                                                      \
            ::grav_test::FailAssertion(#expr, __FILE__, __LINE__, _grav_oss.str());                \
        }                                                                                          \
    } while (false)

#define GRAV_ASSERT_NEAR(val, expected, tol)                                                       \
    do                                                                                             \
    {                                                                                              \
        const double _v = static_cast<double>(val);                                                \
        const double _e = static_cast<double>(expected);                                           \
        const double _t = static_cast<double>(tol);                                                \
        if (std::abs(_v - _e) > _t)                                                                \
        {                                                                                          \
            std::ostringstream _grav_oss;                                                          \
            _grav_oss << "got " << _v << ", expected " << _e << " (diff " << std::abs(_v - _e)     \
                      << " > tol " << _t << ")";                                                   \
            ::grav_test::FailAssertion(#val " ~= " #expected, __FILE__, __LINE__,                  \
                                       _grav_oss.str());                                           \
        }                                                                                          \
    } while (false)

#define GRAV_ASSERT_NO_GL_ERROR(label) ::grav_test::AssertNoGlError((label), __FILE__, __LINE__)
