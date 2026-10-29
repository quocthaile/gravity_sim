#include "render/shader_cache.hpp"

#include "render/asset_paths.hpp"
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace grav_render
{
namespace
{
std::string ReadTextFileResolved(const std::string &relativeOrAbsolutePath)
{
    const std::filesystem::path resolved = ResolveAssetPath(relativeOrAbsolutePath);
    std::ifstream file(resolved, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open shader file: " + resolved.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
} // namespace

ShaderCache::~ShaderCache()
{
    // Note: Shutdown() should be called explicitly before OpenGL context destruction.
}

GLuint ShaderCache::CompileSingleStage(GLenum stage, const std::string &debugLabel,
                                       const std::string &source)
{
    GLuint shader = glCreateShader(stage);
    if (shader == 0)
    {
        throw std::runtime_error("glCreateShader failed for " + debugLabel);
    }

    const char *srcPtr = source.c_str();
    glShaderSource(shader, 1, &srcPtr, nullptr);
    glCompileShader(shader);

    GLint success = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success != GL_TRUE)
    {
        GLint logLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
        std::string infoLog(static_cast<std::size_t>(logLen > 0 ? logLen : 1), '\0');
        if (logLen > 0)
        {
            glGetShaderInfoLog(shader, logLen, nullptr, infoLog.data());
        }
        glDeleteShader(shader);
        throw std::runtime_error("Shader compilation failed (" + debugLabel + "): " + infoLog);
    }
    return shader;
}

GLuint ShaderCache::CompileAndLinkGraphicsFromSource(const std::string &debugLabel,
                                                     const std::string &vertexSource,
                                                     const std::string &fragmentSource)
{
    const GLuint vs = CompileSingleStage(GL_VERTEX_SHADER, debugLabel + ":VS", vertexSource);
    GLuint fs = 0;
    try
    {
        fs = CompileSingleStage(GL_FRAGMENT_SHADER, debugLabel + ":FS", fragmentSource);
    }
    catch (...)
    {
        glDeleteShader(vs);
        throw;
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint success = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success != GL_TRUE)
    {
        GLint logLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
        std::string infoLog(static_cast<std::size_t>(logLen > 0 ? logLen : 1), '\0');
        if (logLen > 0)
        {
            glGetProgramInfoLog(program, logLen, nullptr, infoLog.data());
        }
        glDeleteProgram(program);
        throw std::runtime_error("Graphics program link failed (" + debugLabel + "): " + infoLog);
    }
    return program;
}

GLuint ShaderCache::CompileAndLinkComputeFromSource(const std::string &debugLabel,
                                                    const std::string &computeSource)
{
    const GLuint cs = CompileSingleStage(GL_COMPUTE_SHADER, debugLabel + ":CS", computeSource);
    const GLuint program = glCreateProgram();
    glAttachShader(program, cs);
    glLinkProgram(program);
    glDeleteShader(cs);

    GLint success = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success != GL_TRUE)
    {
        GLint logLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
        std::string infoLog(static_cast<std::size_t>(logLen > 0 ? logLen : 1), '\0');
        if (logLen > 0)
        {
            glGetProgramInfoLog(program, logLen, nullptr, infoLog.data());
        }
        glDeleteProgram(program);
        throw std::runtime_error("Compute program link failed (" + debugLabel + "): " + infoLog);
    }
    return program;
}

GLuint ShaderCache::GetOrCreateGraphics(const std::string &key,
                                        const std::string &vertexRelativePath,
                                        const std::string &fragmentRelativePath,
                                        const std::string &ownerTag)
{
    const auto it = programs_.find(key);
    if (it != programs_.end())
    {
        return it->second.program;
    }

    const std::string vsSrc = ReadTextFileResolved(vertexRelativePath);
    const std::string fsSrc = ReadTextFileResolved(fragmentRelativePath);
    const GLuint program = CompileAndLinkGraphicsFromSource(key, vsSrc, fsSrc);

    CachedEntry entry;
    entry.program = program;
    if (census_ != nullptr)
    {
        const std::size_t approxBytes = vsSrc.size() + fsSrc.size();
        entry.censusId = census_->Register(ResourceCategory::ShaderProgram, program, approxBytes,
                                           approxBytes, ownerTag, key);
    }
    programs_.emplace(key, entry);
    return program;
}

GLuint ShaderCache::GetOrCreateGraphicsFromSource(const std::string &key,
                                                  const std::string &vertexSource,
                                                  const std::string &fragmentSource,
                                                  const std::string &ownerTag)
{
    const auto it = programs_.find(key);
    if (it != programs_.end())
    {
        return it->second.program;
    }

    const GLuint program = CompileAndLinkGraphicsFromSource(key, vertexSource, fragmentSource);
    CachedEntry entry;
    entry.program = program;
    if (census_ != nullptr)
    {
        const std::size_t approxBytes = vertexSource.size() + fragmentSource.size();
        entry.censusId = census_->Register(ResourceCategory::ShaderProgram, program, approxBytes,
                                           approxBytes, ownerTag, key);
    }
    programs_.emplace(key, entry);
    return program;
}

GLuint ShaderCache::GetOrCreateCompute(const std::string &key,
                                       const std::string &computeRelativePath,
                                       const std::string &ownerTag)
{
    const auto it = programs_.find(key);
    if (it != programs_.end())
    {
        return it->second.program;
    }

    const std::string csSrc = ReadTextFileResolved(computeRelativePath);
    const GLuint program = CompileAndLinkComputeFromSource(key, csSrc);

    CachedEntry entry;
    entry.program = program;
    if (census_ != nullptr)
    {
        const std::size_t approxBytes = csSrc.size();
        entry.censusId = census_->Register(ResourceCategory::ShaderProgram, program, approxBytes,
                                           approxBytes, ownerTag, key);
    }
    programs_.emplace(key, entry);
    return program;
}

bool ShaderCache::Contains(const std::string &key) const { return programs_.contains(key); }

void ShaderCache::Invalidate(const std::string &key)
{
    const auto it = programs_.find(key);
    if (it == programs_.end())
    {
        return;
    }
    if (census_ != nullptr && it->second.censusId != 0)
    {
        census_->Unregister(it->second.censusId);
    }
    if (it->second.program != 0)
    {
        glDeleteProgram(it->second.program);
    }
    programs_.erase(it);
}

void ShaderCache::Shutdown()
{
    for (auto &[_, entry] : programs_)
    {
        if (census_ != nullptr && entry.censusId != 0)
        {
            census_->Unregister(entry.censusId);
        }
        if (entry.program != 0)
        {
            glDeleteProgram(entry.program);
            entry.program = 0;
        }
    }
    programs_.clear();
}

} // namespace grav_render

