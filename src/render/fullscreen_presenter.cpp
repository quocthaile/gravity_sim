#include "render/fullscreen_presenter.hpp"

namespace grav_render
{

void FullscreenPresenter::Initialize(ShaderCache &shaderCache, ResourceCensus &census)
{
    if (IsInitialized())
    {
        return;
    }

    program_ = shaderCache.GetOrCreateGraphics(
        "shared:fullscreen_quad", "shaders/fullscreen_quad.vert", "shaders/fullscreen_quad.frag",
        "shared:fullscreen_quad");
    locScreenTexture_ = glGetUniformLocation(program_, "u_screenTexture");
    locFlipY_ = glGetUniformLocation(program_, "u_flipY");

    constexpr float kQuadVertices[] = {
        // pos (x, y)   // uv (u, v)
        -1.0f,  1.0f,   0.0f, 1.0f,
        -1.0f, -1.0f,   0.0f, 0.0f,
         1.0f, -1.0f,   1.0f, 0.0f,

        -1.0f,  1.0f,   0.0f, 1.0f,
         1.0f, -1.0f,   1.0f, 0.0f,
         1.0f,  1.0f,   1.0f, 1.0f,
    };

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadVertices), kQuadVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          reinterpret_cast<const void *>(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    vaoCensusId_ = census.Register(ResourceCategory::VertexArray, vao_, 0, 0,
                                   "shared:fullscreen_quad", "FullscreenQuadVAO");
    vboCensusId_ = census.Register(ResourceCategory::GeometryBuffer, vbo_, sizeof(kQuadVertices),
                                   sizeof(kQuadVertices), "shared:fullscreen_quad",
                                   "FullscreenQuadVBO");
}

void FullscreenPresenter::DrawTexture(GLuint texture2D, bool flipY) const
{
    if (!IsInitialized() || texture2D == 0)
    {
        return;
    }

    const GLboolean depthWasEnabled = glIsEnabled(GL_DEPTH_TEST);
    glDisable(GL_DEPTH_TEST);

    glUseProgram(program_);
    if (locScreenTexture_ >= 0)
    {
        glUniform1i(locScreenTexture_, 0);
    }
    if (locFlipY_ >= 0)
    {
        glUniform1i(locFlipY_, flipY ? 1 : 0);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture2D);
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (depthWasEnabled == GL_TRUE)
    {
        glEnable(GL_DEPTH_TEST);
    }
}

void FullscreenPresenter::Shutdown(ResourceCensus *census)
{
    if (census != nullptr)
    {
        census->Unregister(vaoCensusId_);
        census->Unregister(vboCensusId_);
    }
    if (vao_ != 0)
    {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (vbo_ != 0)
    {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    program_ = 0;
    locScreenTexture_ = -1;
    locFlipY_ = -1;
}

} // namespace grav_render

