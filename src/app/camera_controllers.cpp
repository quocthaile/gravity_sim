#include "app/camera_controllers.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace grav_app
{

void FreeFlyCamera::ConfigureDefaults(const glm::vec3 &pos, float yawDeg, float pitchDeg,
                                      float moveSpeed, float scrollSpeed, float fovDeg, float nearZ,
                                      float farZ)
{
    defaultPosition_ = pos;
    defaultYawDeg_ = yawDeg;
    defaultPitchDeg_ = pitchDeg;
    moveSpeed_ = moveSpeed;
    scrollSpeed_ = scrollSpeed;
    fovDeg_ = fovDeg;
    nearZ_ = nearZ;
    farZ_ = farZ;
    Reset();
}

void FreeFlyCamera::Reset()
{
    position_ = defaultPosition_;
    yawDeg_ = defaultYawDeg_;
    pitchDeg_ = defaultPitchDeg_;
    up_ = glm::vec3(0.0f, 1.0f, 0.0f);
    rotating_ = false;
    RecomputeBasis();
    ++revision_;
}

void FreeFlyCamera::RecomputeBasis()
{
    pitchDeg_ = std::clamp(pitchDeg_, -89.0f, 89.0f);
    glm::vec3 f;
    f.x = std::cos(glm::radians(yawDeg_)) * std::cos(glm::radians(pitchDeg_));
    f.y = std::sin(glm::radians(pitchDeg_));
    f.z = std::sin(glm::radians(yawDeg_)) * std::cos(glm::radians(pitchDeg_));
    front_ = glm::normalize(f);
}

bool FreeFlyCamera::HandleInput(const grav_sim::ModuleInputState &input, double wallDeltaSeconds,
                                bool rightDragRotates)
{
    bool changed = false;
    if (input.focusLostThisFrame)
    {
        rotating_ = false;
    }
    if (input.resetCameraRequested)
    {
        Reset();
        return true;
    }

    const float dtSec = static_cast<float>(std::clamp(wallDeltaSeconds, 0.0, 0.25));
    const float step = moveSpeed_ * dtSec;
    const glm::vec3 right = glm::normalize(glm::cross(front_, up_));

    if (input.moveForward)
    {
        position_ += step * front_;
        changed = true;
    }
    if (input.moveBackward)
    {
        position_ -= step * front_;
        changed = true;
    }
    if (input.moveLeft)
    {
        position_ -= step * right;
        changed = true;
    }
    if (input.moveRight)
    {
        position_ += step * right;
        changed = true;
    }
    if (input.moveUp)
    {
        position_ += step * up_;
        changed = true;
    }
    if (input.moveDown)
    {
        position_ -= step * up_;
        changed = true;
    }

    if (rightDragRotates)
    {
        if (input.mouseRightJustPressed)
        {
            rotating_ = true;
        }
        if (!input.mouseRightDown || input.mouseRightJustReleased)
        {
            rotating_ = false;
        }
        if (rotating_ && (std::fabs(input.mouseDeltaX) > 0.0 || std::fabs(input.mouseDeltaY) > 0.0))
        {
            yawDeg_ += static_cast<float>(input.mouseDeltaX) * sensitivity_;
            pitchDeg_ -= static_cast<float>(input.mouseDeltaY) * sensitivity_;
            RecomputeBasis();
            changed = true;
        }
    }

    if (std::fabs(input.scrollDeltaY) > 1e-5)
    {
        position_ += static_cast<float>(input.scrollDeltaY) * scrollSpeed_ * front_;
        changed = true;
    }

    if (changed)
    {
        ++revision_;
    }
    return changed;
}

glm::mat4 FreeFlyCamera::ViewMatrix() const
{
    return glm::lookAt(position_, position_ + front_, up_);
}

glm::mat4 FreeFlyCamera::ProjectionMatrix(float aspect) const
{
    const float safeAspect = aspect > 0.01f ? aspect : (16.0f / 9.0f);
    return glm::perspective(glm::radians(fovDeg_), safeAspect, nearZ_, farZ_);
}

void OrbitPanCamera::ConfigureDefaults(const glm::vec3 &target, float radius, float minRadius,
                                       float maxRadius, float azimuthRad, float elevationRad,
                                       float fovDeg, bool allowPan)
{
    defaultTarget_ = target;
    defaultRadius_ = radius;
    minRadius_ = minRadius;
    maxRadius_ = maxRadius;
    defaultAzimuthRad_ = azimuthRad;
    defaultElevationRad_ = elevationRad;
    fovDeg_ = fovDeg;
    allowPan_ = allowPan;
    Reset();
}

void OrbitPanCamera::Reset()
{
    target_ = defaultTarget_;
    radius_ = std::clamp(defaultRadius_, minRadius_, maxRadius_);
    azimuthRad_ = defaultAzimuthRad_;
    elevationRad_ = std::clamp(defaultElevationRad_, 0.01f, glm::pi<float>() - 0.01f);
    dragging_ = false;
    panning_ = false;
    ++revision_;
}

bool OrbitPanCamera::HandleInput(const grav_sim::ModuleInputState &input)
{
    bool changed = false;
    if (input.focusLostThisFrame)
    {
        dragging_ = false;
        panning_ = false;
    }
    if (input.resetCameraRequested)
    {
        Reset();
        return true;
    }

    if (input.mouseLeftJustPressed || input.mouseMiddleJustPressed)
    {
        dragging_ = true;
        panning_ = allowPan_ && (input.mouseMiddleDown || input.shiftHeld);
    }
    if (!input.mouseLeftDown && !input.mouseMiddleDown)
    {
        dragging_ = false;
        panning_ = false;
    }

    if (dragging_ && (std::fabs(input.mouseDeltaX) > 0.0 || std::fabs(input.mouseDeltaY) > 0.0))
    {
        const float dx = static_cast<float>(input.mouseDeltaX);
        const float dy = static_cast<float>(input.mouseDeltaY);
        if (panning_)
        {
            glm::vec3 pos, right, up, fwd;
            CameraBasis(pos, right, up, fwd);
            const float panScale = 0.0012f * radius_;
            target_ += (-right * dx + up * dy) * panScale;
        }
        else
        {
            constexpr float kOrbitSpeed = 0.008f;
            azimuthRad_ += dx * kOrbitSpeed;
            elevationRad_ =
                std::clamp(elevationRad_ - dy * kOrbitSpeed, 0.01f, glm::pi<float>() - 0.01f);
        }
        changed = true;
    }

    if (std::fabs(input.scrollDeltaY) > 1e-5)
    {
        constexpr float kZoomFactor = 1.09f;
        if (input.scrollDeltaY > 0.0)
        {
            radius_ /= std::pow(kZoomFactor, static_cast<float>(input.scrollDeltaY));
        }
        else
        {
            radius_ *= std::pow(kZoomFactor, static_cast<float>(-input.scrollDeltaY));
        }
        radius_ = std::clamp(radius_, minRadius_, maxRadius_);
        changed = true;
    }

    if (changed)
    {
        ++revision_;
    }
    return changed;
}

glm::vec3 OrbitPanCamera::Position() const
{
    const float clampedEl = std::clamp(elevationRad_, 0.01f, glm::pi<float>() - 0.01f);
    return glm::vec3(target_.x + radius_ * std::sin(clampedEl) * std::cos(azimuthRad_),
                     target_.y + radius_ * std::cos(clampedEl),
                     target_.z + radius_ * std::sin(clampedEl) * std::sin(azimuthRad_));
}

glm::vec3 OrbitPanCamera::Forward() const
{
    const glm::vec3 pos = Position();
    const glm::vec3 diff = target_ - pos;
    const float len = glm::length(diff);
    return len > 1e-6f ? (diff / len) : glm::vec3(0.0f, 0.0f, -1.0f);
}

void OrbitPanCamera::CameraBasis(glm::vec3 &outPos, glm::vec3 &outRight, glm::vec3 &outUp,
                                 glm::vec3 &outForward) const
{
    outPos = Position();
    outForward = Forward();
    const glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
    glm::vec3 rightCandidate = glm::cross(outForward, worldUp);
    if (glm::dot(rightCandidate, rightCandidate) < 1e-8f)
    {
        rightCandidate = glm::cross(outForward, glm::vec3(0.0f, 0.0f, 1.0f));
    }
    outRight = glm::normalize(rightCandidate);
    outUp = glm::normalize(glm::cross(outRight, outForward));
}

glm::mat4 OrbitPanCamera::ViewMatrix(float worldScaleDivisor) const
{
    const float invScale = worldScaleDivisor != 0.0f ? (1.0f / worldScaleDivisor) : 1.0f;
    glm::vec3 pos, right, up, fwd;
    CameraBasis(pos, right, up, fwd);
    const glm::vec3 scaledPos = pos * invScale;
    const glm::vec3 scaledTarget = target_ * invScale;
    return glm::lookAt(scaledPos, scaledTarget, up);
}

glm::mat4 OrbitPanCamera::ProjectionMatrix(float aspect, float nearZ, float farZ) const
{
    const float safeAspect = aspect > 0.01f ? aspect : (16.0f / 9.0f);
    return glm::perspective(glm::radians(fovDeg_), safeAspect, nearZ, farZ);
}

void OrbitPanCamera::SetTarget(const glm::vec3 &target)
{
    if (glm::length(target - target_) > 1e-5f)
    {
        target_ = target;
        ++revision_;
    }
}

void OrbitPanCamera::SetFovDegrees(float fovDeg)
{
    const float clamped = std::clamp(fovDeg, 15.0f, 120.0f);
    if (std::fabs(clamped - fovDeg_) > 1e-4f)
    {
        fovDeg_ = clamped;
        ++revision_;
    }
}

void PlanarOrthoCamera::ConfigureDefaults(float halfWidthWorld, float minHalfWidth,
                                          float maxHalfWidth)
{
    defaultHalfWidthWorld_ = halfWidthWorld;
    minHalfWidth_ = minHalfWidth;
    maxHalfWidth_ = maxHalfWidth;
    Reset();
}

void PlanarOrthoCamera::Reset()
{
    halfWidthWorld_ = std::clamp(defaultHalfWidthWorld_, minHalfWidth_, maxHalfWidth_);
    centerXz_ = glm::vec2(0.0f, 0.0f);
    panning_ = false;
    ++revision_;
}

bool PlanarOrthoCamera::HandleInput(const grav_sim::ModuleInputState &input, int viewportWidth,
                                    int viewportHeight)
{
    bool changed = false;
    if (input.focusLostThisFrame)
    {
        panning_ = false;
    }
    if (input.resetCameraRequested)
    {
        Reset();
        return true;
    }

    if (input.mouseMiddleJustPressed || (input.mouseLeftJustPressed && !input.shiftHeld))
    {
        panning_ = true;
    }
    if (!input.mouseMiddleDown && !input.mouseLeftDown)
    {
        panning_ = false;
    }

    if (panning_ && (std::fabs(input.mouseDeltaX) > 0.0 || std::fabs(input.mouseDeltaY) > 0.0))
    {
        const float safeW = static_cast<float>(std::max(viewportWidth, 1));
        const float safeH = static_cast<float>(std::max(viewportHeight, 1));
        const float aspect = safeW / safeH;
        const float halfHeightWorld = halfWidthWorld_ / aspect;

        const float worldPerPixelX = (2.0f * halfWidthWorld_) / safeW;
        const float worldPerPixelZ = (2.0f * halfHeightWorld) / safeH;

        centerXz_.x -= static_cast<float>(input.mouseDeltaX) * worldPerPixelX;
        // Screen +Y goes down; in our top-down lookAt((x,H,z), (x,0,z), up=(0,0,-1)),
        // screen up corresponds to -Z (or +Y in 2D_lensing if Z = -y2d or +y2d).
        // With up=(0, 0, 1) looking from (0, +H, 0) to (0, 0, 0), wait: right = cross((0,-1,0), (0,0,1)) = (-1,0,0)!
        // With up=(0, 0, -1), right = cross((0,-1,0), (0,0,-1)) = (+1, 0, 0), screen +Y (top) is -Z,
        // or with up=(0, 0, 1) looking from (0, -H, 0)? Let's match ViewMatrix() up=(0, 0, 1) from +Y?
        // Wait: cross((0,-1,0), (0,0,-1)) = (+1, 0, 0) so +X is right and -Z is up (so Z_world = -Y_2D or Z_world = +Y_2D).
        // Let's keep Z_world = -Y_screen so dragging mouse down (deltaY > 0) moves view up (-Z).
        centerXz_.y -= static_cast<float>(input.mouseDeltaY) * worldPerPixelZ;
        changed = true;
    }

    if (std::fabs(input.scrollDeltaY) > 1e-5)
    {
        constexpr float kZoomFactor = 1.10f;
        if (input.scrollDeltaY > 0.0)
        {
            halfWidthWorld_ /= std::pow(kZoomFactor, static_cast<float>(input.scrollDeltaY));
        }
        else
        {
            halfWidthWorld_ *= std::pow(kZoomFactor, static_cast<float>(-input.scrollDeltaY));
        }
        halfWidthWorld_ = std::clamp(halfWidthWorld_, minHalfWidth_, maxHalfWidth_);
        changed = true;
    }

    if (changed)
    {
        ++revision_;
    }
    return changed;
}

glm::mat4 PlanarOrthoCamera::ViewMatrix() const
{
    // Looking down from +Y toward XZ plane; up = (0, 0, -1) gives +X to the right and -Z upward,
    // or if we map 2D (x, y) -> 3D (x, 0, -y) (or look from -Y? No, from +Y we see the grid funnel below!).
    // Wait: if we map 2D (x, y) -> 3D (x, 0, z = y) and want +y_2D at the TOP of the screen in 2D ortho
    // AND +x_2D to the RIGHT:
    // If eye = (cx, +500, cz), center = (cx, 0, cz), forward = (0, -1, 0):
    // For right = (+1, 0, 0), up = cross(right, forward) = cross((1,0,0), (0,-1,0)) = (0, 0, -1).
    // Then +Z points downward on screen. If instead we map planar (x, y) -> (x, 0, z = -y) OR flip ortho top/bottom:
    // Keeping eye = (cx, +500, cz) with up = (0, 0, 1) makes right = (-1, 0, 0).
    // Better: eye = (cx, +500, cz), up = (0, 0, -1) so +X is right, and in ProjectionMatrix use (-halfH, +halfH)
    // while mapping planar (x, y_2d) -> 3D (x, 0, z = -y_2d) OR (x, 0, z = y_2d) with documented sign!
    // Even simpler: in ProjectionMatrix we can set bottom = +halfH, top = -halfH? Wait: flipping ortho flips winding!
    // Mapping planar (x_2d, y_2d) -> 3D (x = x_2d, y = 0, z = -y_2d) preserves right-handed 3D coordinates:
    // (+X_3D = East/+x_2d, +Y_3D = Up/Zenith, -Z_3D = North/+y_2d)!
    // That is the standard OpenGL right-handed convention! Let's check panning: when mouse drags right (deltaX > 0),
    // centerXz_.x decreases; when mouse drags down (deltaY > 0), screen moves down so centerXz_.y (-Z is up) decreases!
    const glm::vec3 eye(centerXz_.x, 500.0f, centerXz_.y);
    const glm::vec3 center(centerXz_.x, 0.0f, centerXz_.y);
    return glm::lookAt(eye, center, glm::vec3(0.0f, 0.0f, -1.0f));
}

glm::mat4 PlanarOrthoCamera::ProjectionMatrix(float aspect) const
{
    const float safeAspect = aspect > 0.01f ? aspect : (16.0f / 9.0f);
    const float halfH = halfWidthWorld_ / safeAspect;
    return glm::ortho(-halfWidthWorld_, halfWidthWorld_, -halfH, halfH, 0.1f, 2000.0f);
}

} // namespace grav_app

