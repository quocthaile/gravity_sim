#pragma once

#include "render/resource_census.hpp"
#include "render/shader_cache.hpp"
#include <GL/glew.h>
#include <cstdint>

namespace grav_render
{

class FullscreenPresenter
{
  public:
    FullscreenPresenter() = default;
    ~FullscreenPresenter() = default;

    FullscreenPresenter(const FullscreenPresenter &) = delete;
    FullscreenPresenter &operator=(const FullscreenPresenter &) = delete;

    void Initialize(ShaderCache &shaderCache, ResourceCensus &census);
    void DrawTexture(GLuint texture2D, bool flipY = false) const;
    void Shutdown(ResourceCensus *census = nullptr);

    bool IsInitialized() const { return vao_ != 0 && program_ != 0; }
    GLuint VaoHandle() const { return vao_; }

  private:
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLuint program_ = 0;
    GLint locScreenTexture_ = -1;
    GLint locFlipY_ = -1;
    std::uint64_t vaoCensusId_ = 0;
    std::uint64_t vboCensusId_ = 0;
};

} // namespace grav_render

