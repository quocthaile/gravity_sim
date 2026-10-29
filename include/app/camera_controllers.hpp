#pragma once

#include "models/simulation_module.hpp"
#include <cstdint>
#include <glm/glm.hpp>

namespace grav_app
{

class FreeFlyCamera
{
  public:
    void ConfigureDefaults(const glm::vec3 &pos, float yawDeg, float pitchDeg, float moveSpeed,
                           float scrollSpeed, float fovDeg = 45.0f, float nearZ = 0.1f,
                           float farZ = 750000.0f);
    void Reset();
    bool HandleInput(const grav_sim::ModuleInputState &input, double wallDeltaSeconds,
                     bool rightDragRotates = true);

    glm::mat4 ViewMatrix() const;
    glm::mat4 ProjectionMatrix(float aspect) const;

    const glm::vec3 &Position() const { return position_; }
    const glm::vec3 &Front() const { return front_; }
    const glm::vec3 &Up() const { return up_; }
    float FovDegrees() const { return fovDeg_; }
    std::uint64_t Revision() const { return revision_; }
    void BumpRevision() { ++revision_; }

  private:
    void RecomputeBasis();

    glm::vec3 defaultPosition_{0.0f, 1000.0f, 5000.0f};
    float defaultYawDeg_ = -90.0f;
    float defaultPitchDeg_ = 0.0f;

    glm::vec3 position_{0.0f, 1000.0f, 5000.0f};
    glm::vec3 front_{0.0f, 0.0f, -1.0f};
    glm::vec3 up_{0.0f, 1.0f, 0.0f};
    float yawDeg_ = -90.0f;
    float pitchDeg_ = 0.0f;

    float moveSpeed_ = 1000.0f;
    float scrollSpeed_ = 1200.0f;
    float sensitivity_ = 0.12f;
    float fovDeg_ = 45.0f;
    float nearZ_ = 0.1f;
    float farZ_ = 750000.0f;

    bool rotating_ = false;
    std::uint64_t revision_ = 1;
};

class OrbitPanCamera
{
  public:
    void ConfigureDefaults(const glm::vec3 &target, float radius, float minRadius, float maxRadius,
                           float azimuthRad, float elevationRad, float fovDeg = 60.0f,
                           bool allowPan = true);
    void Reset();
    bool HandleInput(const grav_sim::ModuleInputState &input);

    glm::vec3 Position() const;
    glm::vec3 Forward() const;
    void CameraBasis(glm::vec3 &outPos, glm::vec3 &outRight, glm::vec3 &outUp,
                     glm::vec3 &outForward) const;

    glm::mat4 ViewMatrix(float worldScaleDivisor = 1.0f) const;
    glm::mat4 ProjectionMatrix(float aspect, float nearZ, float farZ) const;

    const glm::vec3 &Target() const { return target_; }
    void SetTarget(const glm::vec3 &target);
    float Radius() const { return radius_; }
    float FovDegrees() const { return fovDeg_; }
    void SetFovDegrees(float fovDeg);
    bool IsMoving() const { return dragging_; }
    std::uint64_t Revision() const { return revision_; }

  private:
    glm::vec3 defaultTarget_{0.0f};
    float defaultRadius_ = 6.34194e10f;
    float defaultAzimuthRad_ = 0.0f;
    float defaultElevationRad_ = 1.5707963f;

    glm::vec3 target_{0.0f};
    float radius_ = 6.34194e10f;
    float minRadius_ = 1.5e10f;
    float maxRadius_ = 1.0e12f;
    float azimuthRad_ = 0.0f;
    float elevationRad_ = 1.5707963f;
    float fovDeg_ = 60.0f;
    bool allowPan_ = true;

    bool dragging_ = false;
    bool panning_ = false;
    std::uint64_t revision_ = 1;
};

class PlanarOrthoCamera
{
  public:
    void ConfigureDefaults(float halfWidthWorld, float minHalfWidth, float maxHalfWidth);
    void Reset();
    bool HandleInput(const grav_sim::ModuleInputState &input, int viewportWidth,
                     int viewportHeight);

    // Maps 3D XZ world (x, y, z) onto a top-down orthographic view looking along -Y.
    glm::mat4 ViewMatrix() const;
    glm::mat4 ProjectionMatrix(float aspect) const;

    float HalfWidthWorld() const { return halfWidthWorld_; }
    const glm::vec2 &CenterXz() const { return centerXz_; }
    std::uint64_t Revision() const { return revision_; }

  private:
    float defaultHalfWidthWorld_ = 120.0f;
    float minHalfWidth_ = 5.0f;
    float maxHalfWidth_ = 2500.0f;

    float halfWidthWorld_ = 120.0f;
    glm::vec2 centerXz_{0.0f, 0.0f};
    bool panning_ = false;
    std::uint64_t revision_ = 1;
};

} // namespace grav_app

