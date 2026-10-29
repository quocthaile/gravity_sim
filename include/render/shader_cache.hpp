#pragma once

#include "render/resource_census.hpp"
#include <GL/glew.h>
#include <cstdint>
#include <map>
#include <string>

namespace grav_render
{

class ShaderCache
{
  public:
    explicit ShaderCache(ResourceCensus *census = nullptr) : census_(census) {}
    ~ShaderCache();

    ShaderCache(const ShaderCache &) = delete;
    ShaderCache &operator=(const ShaderCache &) = delete;

    void SetCensus(ResourceCensus *census) { census_ = census; }

    GLuint GetOrCreateGraphics(const std::string &key, const std::string &vertexRelativePath,
                               const std::string &fragmentRelativePath,
                               const std::string &ownerTag = "shared:shader_cache");

    GLuint GetOrCreateGraphicsFromSource(const std::string &key, const std::string &vertexSource,
                                         const std::string &fragmentSource,
                                         const std::string &ownerTag = "shared:shader_cache");

    GLuint GetOrCreateCompute(const std::string &key, const std::string &computeRelativePath,
                              const std::string &ownerTag = "shared:shader_cache");

    bool Contains(const std::string &key) const;
    void Invalidate(const std::string &key);
    void Shutdown();

    std::size_t CachedProgramCount() const { return programs_.size(); }

    static GLuint CompileAndLinkGraphicsFromSource(const std::string &debugLabel,
                                                   const std::string &vertexSource,
                                                   const std::string &fragmentSource);
    static GLuint CompileAndLinkComputeFromSource(const std::string &debugLabel,
                                                  const std::string &computeSource);

  private:
    struct CachedEntry
    {
        GLuint program = 0;
        std::uint64_t censusId = 0;
    };

    static GLuint CompileSingleStage(GLenum stage, const std::string &debugLabel,
                                     const std::string &source);

    ResourceCensus *census_ = nullptr;
    std::map<std::string, CachedEntry> programs_;
};

} // namespace grav_render

