#include "models/black_hole/black_hole_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace grav_models
{
namespace
{

glm::dvec2 AccelCanonical2D(const glm::dvec2 &uPos, const glm::dvec2 &uVel)
{
    const double r2 = std::max(glm::dot(uPos, uPos), 1e-12);
    const double r = std::sqrt(r2);
    const double r5 = r2 * r2 * r;
    const double h = uPos.x * uVel.y - uPos.y * uVel.x;
    return (-1.5 * (h * h) / r5) * uPos;
}

void Rk4StepCanonical2D(glm::dvec2 &uPos, glm::dvec2 &uVel, double h)
{
    const glm::dvec2 k1x = uVel;
    const glm::dvec2 k1v = AccelCanonical2D(uPos, uVel);

    const glm::dvec2 p2 = uPos + 0.5 * h * k1x;
    const glm::dvec2 v2 = uVel + 0.5 * h * k1v;
    const glm::dvec2 k2x = v2;
    const glm::dvec2 k2v = AccelCanonical2D(p2, v2);

    const glm::dvec2 p3 = uPos + 0.5 * h * k2x;
    const glm::dvec2 v3 = uVel + 0.5 * h * k2v;
    const glm::dvec2 k3x = v3;
    const glm::dvec2 k3v = AccelCanonical2D(p3, v3);

    const glm::dvec2 p4 = uPos + h * k3x;
    const glm::dvec2 v4 = uVel + h * k3v;
    const glm::dvec2 k4x = v4;
    const glm::dvec2 k4v = AccelCanonical2D(p4, v4);

    uPos += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    uVel += (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
}

// Exact RHS from 2D_lensing.cpp:256-298
void LegacyPlanarGeodesicRhs(double r, double dr, double dphi, double E, double rs, double rhs[4])
{
    const double safeR = std::max(r, rs * 1.000001);
    const double f = std::max(1.0 - rs / safeR, 1e-9);
    const double dt_dLambda = E / f;

    rhs[0] = dr;
    rhs[1] = dphi;
    rhs[2] = -(rs / (2.0 * safeR * safeR)) * f * (dt_dLambda * dt_dLambda) +
             (rs / (2.0 * safeR * safeR * f)) * (dr * dr) + (safeR - rs) * (dphi * dphi);
    rhs[3] = -2.0 * dr * dphi / safeR;
}

glm::dvec3 AccelCanonical3D(const glm::dvec3 &uPos, const glm::dvec3 &uVel)
{
    const double r2 = std::max(glm::dot(uPos, uPos), 1e-12);
    const double r = std::sqrt(r2);
    const double r5 = r2 * r2 * r;
    const glm::dvec3 hVec = glm::cross(uPos, uVel);
    const double h2 = glm::dot(hVec, hVec);
    return (-1.5 * h2 / r5) * uPos;
}

void Rk4StepCanonical3D(glm::dvec3 &uPos, glm::dvec3 &uVel, double h)
{
    const glm::dvec3 k1x = uVel;
    const glm::dvec3 k1v = AccelCanonical3D(uPos, uVel);

    const glm::dvec3 p2 = uPos + 0.5 * h * k1x;
    const glm::dvec3 v2 = uVel + 0.5 * h * k1v;
    const glm::dvec3 k2x = v2;
    const glm::dvec3 k2v = AccelCanonical3D(p2, v2);

    const glm::dvec3 p3 = uPos + 0.5 * h * k2x;
    const glm::dvec3 v3 = uVel + 0.5 * h * k2v;
    const glm::dvec3 k3x = v3;
    const glm::dvec3 k3v = AccelCanonical3D(p3, v3);

    const glm::dvec3 p4 = uPos + h * k3x;
    const glm::dvec3 v4 = uVel + h * k3v;
    const glm::dvec3 k4x = v4;
    const glm::dvec3 k4v = AccelCanonical3D(p4, v4);

    uPos += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    uVel += (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
}

bool SegmentHitsSphereD(const glm::dvec3 &p0, const glm::dvec3 &p1, const glm::dvec3 &center,
                        double radius, double &hitT)
{
    const glm::dvec3 d = p1 - p0;
    const glm::dvec3 oc = p0 - center;
    const double a = glm::dot(d, d);
    if (a < 1e-16)
    {
        hitT = 0.0;
        return glm::length(oc) <= radius;
    }
    const double b = 2.0 * glm::dot(oc, d);
    const double c = glm::dot(oc, oc) - radius * radius;
    const double disc = b * b - 4.0 * a * c;
    if (disc < 0.0)
    {
        hitT = 1.0;
        return false;
    }
    const double s = std::sqrt(disc);
    const double t0 = (-b - s) / (2.0 * a);
    const double t1 = (-b + s) / (2.0 * a);
    if (t0 >= 0.0 && t0 <= 1.0)
    {
        hitT = t0;
        return true;
    }
    if (t1 >= 0.0 && t1 <= 1.0)
    {
        hitT = t1;
        return true;
    }
    hitT = 1.0;
    return false;
}

struct LegacySphericalRay3D
{
    double x = 0.0, y = 0.0, z = 0.0;
    double r = 0.0, theta = 0.0, phi = 0.0;
    double dr = 0.0, dtheta = 0.0, dphi = 0.0;
    double E = 0.0, L = 0.0;
};

// Matches geodesic.comp:43 & CPU-geodesic.cpp:374
LegacySphericalRay3D InitLegacySphericalRay3D(const glm::dvec3 &relPos, const glm::dvec3 &dir,
                                              double rs)
{
    LegacySphericalRay3D ray;
    ray.x = relPos.x;
    ray.y = relPos.y;
    ray.z = relPos.z;
    ray.r = std::max(glm::length(relPos), rs * 1.0001);
    const double cosTheta = std::clamp(relPos.z / ray.r, -0.999999999, 0.999999999);
    ray.theta = std::acos(cosTheta);
    ray.phi = std::atan2(relPos.y, relPos.x);

    const double sinT = std::sin(ray.theta);
    const double safeSinT = std::fabs(sinT) < 1e-6 ? (sinT < 0.0 ? -1e-6 : 1e-6) : sinT;
    const double cosT = std::cos(ray.theta);
    const double sinP = std::sin(ray.phi);
    const double cosP = std::cos(ray.phi);

    ray.dr = sinT * cosP * dir.x + sinT * sinP * dir.y + cosT * dir.z;
    ray.dtheta = (cosT * cosP * dir.x + cosT * sinP * dir.y - sinT * dir.z) / ray.r;
    ray.dphi = (-sinP * dir.x + cosP * dir.y) / (ray.r * safeSinT);

    ray.L = ray.r * ray.r * sinT * ray.dphi;
    const double f = std::max(1.0 - rs / ray.r, 1e-8);
    // Preserves legacy source initialization formula from geodesic.comp:58 and CPU-geodesic.cpp:392
    const double dt_dL =
        std::sqrt((ray.dr * ray.dr) / f +
                  ray.r * ray.r * (ray.dtheta * ray.dtheta + sinT * sinT * ray.dphi * ray.dphi));
    ray.E = f * dt_dL;
    return ray;
}

// Matches geodesic.comp:73 & CPU-geodesic.cpp:481
void LegacySphericalGeodesicRhs3D(const LegacySphericalRay3D &ray, double rs, double rhs[6])
{
    const double r = std::max(ray.r, rs * 1.00001);
    const double sinT = std::sin(ray.theta);
    const double cosT = std::cos(ray.theta);
    const double safeSinT = std::fabs(sinT) < 1e-6 ? (sinT < 0.0 ? -1e-6 : 1e-6) : sinT;
    const double f = std::max(1.0 - rs / r, 1e-8);
    const double dt_dL = ray.E / f;

    rhs[0] = ray.dr;
    rhs[1] = ray.dtheta;
    rhs[2] = ray.dphi;
    rhs[3] = -(rs / (2.0 * r * r)) * f * dt_dL * dt_dL + (rs / (2.0 * r * r * f)) * ray.dr * ray.dr +
             r * (ray.dtheta * ray.dtheta + sinT * sinT * ray.dphi * ray.dphi);
    rhs[4] = -(2.0 / r) * ray.dr * ray.dtheta + sinT * cosT * ray.dphi * ray.dphi;
    rhs[5] = -(2.0 / r) * ray.dr * ray.dphi - 2.0 * (cosT / safeSinT) * ray.dtheta * ray.dphi;
}

void StepLegacySphericalEuler3D(LegacySphericalRay3D &ray, double dL, double rs)
{
    double k1[6]{};
    LegacySphericalGeodesicRhs3D(ray, rs, k1);
    ray.r += dL * k1[0];
    ray.theta += dL * k1[1];
    ray.phi += dL * k1[2];
    ray.dr += dL * k1[3];
    ray.dtheta += dL * k1[4];
    ray.dphi += dL * k1[5];

    const double sinT = std::sin(ray.theta);
    ray.x = ray.r * sinT * std::cos(ray.phi);
    ray.y = ray.r * sinT * std::sin(ray.phi);
    ray.z = ray.r * std::cos(ray.theta);
}

void StepLegacySphericalRk43D(LegacySphericalRay3D &ray, double dL, double rs)
{
    const double y0[6] = {ray.r, ray.theta, ray.phi, ray.dr, ray.dtheta, ray.dphi};
    double k1[6]{}, k2[6]{}, k3[6]{}, k4[6]{};

    LegacySphericalGeodesicRhs3D(ray, rs, k1);

    LegacySphericalRay3D r2 = ray;
    r2.r = y0[0] + 0.5 * dL * k1[0];
    r2.theta = y0[1] + 0.5 * dL * k1[1];
    r2.phi = y0[2] + 0.5 * dL * k1[2];
    r2.dr = y0[3] + 0.5 * dL * k1[3];
    r2.dtheta = y0[4] + 0.5 * dL * k1[4];
    r2.dphi = y0[5] + 0.5 * dL * k1[5];
    LegacySphericalGeodesicRhs3D(r2, rs, k2);

    LegacySphericalRay3D r3 = ray;
    r3.r = y0[0] + 0.5 * dL * k2[0];
    r3.theta = y0[1] + 0.5 * dL * k2[1];
    r3.phi = y0[2] + 0.5 * dL * k2[2];
    r3.dr = y0[3] + 0.5 * dL * k2[3];
    r3.dtheta = y0[4] + 0.5 * dL * k2[4];
    r3.dphi = y0[5] + 0.5 * dL * k2[5];
    LegacySphericalGeodesicRhs3D(r3, rs, k3);

    LegacySphericalRay3D r4 = ray;
    r4.r = y0[0] + dL * k3[0];
    r4.theta = y0[1] + dL * k3[1];
    r4.phi = y0[2] + dL * k3[2];
    r4.dr = y0[3] + dL * k3[3];
    r4.dtheta = y0[4] + dL * k3[4];
    r4.dphi = y0[5] + dL * k3[5];
    LegacySphericalGeodesicRhs3D(r4, rs, k4);

    ray.r += (dL / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
    ray.theta += (dL / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
    ray.phi += (dL / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]);
    ray.dr += (dL / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]);
    ray.dtheta += (dL / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]);
    ray.dphi += (dL / 6.0) * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]);

    const double sinT = std::sin(ray.theta);
    ray.x = ray.r * sinT * std::cos(ray.phi);
    ray.y = ray.r * sinT * std::sin(ray.phi);
    ray.z = ray.r * std::cos(ray.theta);
}

} // namespace

void BlackHoleParameters::SetMassKg(double newMassKg)
{
    const double clamped = std::max(newMassKg, 1.0e20);
    if (std::fabs(clamped - massKg) > 0.0)
    {
        massKg = clamped;
        schwarzschildRadiusMeters = ComputeSchwarzschildRadiusMeters(massKg);
        ++revision;
    }
}

void BlackHoleParameters::SetPositionMeters(const glm::dvec3 &newPosMeters)
{
    if (glm::length(newPosMeters - positionMeters) > 0.0)
    {
        positionMeters = newPosMeters;
        ++revision;
    }
}

void BlackHoleParameters::ResetToSagittariusA()
{
    positionMeters = glm::dvec3(0.0, 0.0, 0.0);
    massKg = kSagittariusAMassKg;
    schwarzschildRadiusMeters = ComputeSchwarzschildRadiusMeters(kSagittariusAMassKg);
    ++revision;
}

grav_render::GridSource BlackHoleParameters::ToGridSource(float metersPerWorldUnit) const
{
    const double invUnit = metersPerWorldUnit > 0.0f ? (1.0 / metersPerWorldUnit) : 1.0;
    const glm::vec3 worldPos(static_cast<float>(positionMeters.x * invUnit),
                             static_cast<float>(positionMeters.y * invUnit),
                             static_cast<float>(positionMeters.z * invUnit));
    return grav_render::GridSource{
        glm::vec4(worldPos, static_cast<float>(schwarzschildRadiusMeters))};
}

PlanarRayState InitializePlanarRay(const glm::dvec2 &relPosMeters, const glm::dvec2 &velocityMps,
                                   double rsMeters)
{
    PlanarRayState ray;
    ray.x = relPosMeters.x;
    ray.y = relPosMeters.y;
    ray.r = std::sqrt(ray.x * ray.x + ray.y * ray.y);
    ray.phi = std::atan2(ray.y, ray.x);

    const double safeRs = std::max(rsMeters, 1.0);
    if (ray.r <= safeRs)
    {
        ray.captured = true;
        return ray;
    }

    const double cosP = std::cos(ray.phi);
    const double sinP = std::sin(ray.phi);
    ray.dr = velocityMps.x * cosP + velocityMps.y * sinP;
    ray.dphi = (-velocityMps.x * sinP + velocityMps.y * cosP) / ray.r;
    ray.L = ray.r * ray.r * ray.dphi;

    const double f = 1.0 - safeRs / ray.r;
    const double dt_dL = std::sqrt((ray.dr * ray.dr) / (f * f) + (ray.r * ray.r * ray.dphi * ray.dphi) / f);
    ray.E = f * dt_dL;

    // Dimensionless canonical state u = pos / rs, normalized initial tangent
    const double vMag = glm::length(velocityMps);
    const glm::dvec2 unitDir = vMag > 0.0 ? (velocityMps / vMag) : glm::dvec2(1.0, 0.0);
    ray.uPos = relPosMeters / safeRs;
    ray.uVel = unitDir;
    const double uR = glm::length(ray.uPos);
    ray.uL0 = ray.uPos.x * ray.uVel.y - ray.uPos.y * ray.uVel.x;
    ray.uE0Squared = glm::dot(ray.uVel, ray.uVel) - (ray.uL0 * ray.uL0) / (uR * uR * uR);

    return ray;
}

void StepPlanarRay(PlanarRayState &ray, double dLambdaAffine, double rsMeters,
                   double escapeRadiusMeters, PlanarGeodesicSolverKind solverKind)
{
    if (ray.captured || ray.escaped)
    {
        return;
    }

    const double safeRs = std::max(rsMeters, 1.0);
    if (ray.r <= safeRs)
    {
        ray.captured = true;
        return;
    }

    if (solverKind == PlanarGeodesicSolverKind::LegacyPolarRk4)
    {
        const double y0[4] = {ray.r, ray.phi, ray.dr, ray.dphi};
        double k1[4]{}, k2[4]{}, k3[4]{}, k4[4]{};

        LegacyPlanarGeodesicRhs(y0[0], y0[2], y0[3], ray.E, safeRs, k1);
        LegacyPlanarGeodesicRhs(y0[0] + 0.5 * dLambdaAffine * k1[0],
                                y0[2] + 0.5 * dLambdaAffine * k1[2],
                                y0[3] + 0.5 * dLambdaAffine * k1[3], ray.E, safeRs, k2);
        LegacyPlanarGeodesicRhs(y0[0] + 0.5 * dLambdaAffine * k2[0],
                                y0[2] + 0.5 * dLambdaAffine * k2[2],
                                y0[3] + 0.5 * dLambdaAffine * k2[3], ray.E, safeRs, k3);
        LegacyPlanarGeodesicRhs(y0[0] + dLambdaAffine * k3[0], y0[2] + dLambdaAffine * k3[2],
                                y0[3] + dLambdaAffine * k3[3], ray.E, safeRs, k4);

        ray.r += (dLambdaAffine / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
        ray.phi += (dLambdaAffine / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
        ray.dr += (dLambdaAffine / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]);
        ray.dphi += (dLambdaAffine / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]);

        ray.x = ray.r * std::cos(ray.phi);
        ray.y = ray.r * std::sin(ray.phi);
        ray.uPos = glm::dvec2(ray.x, ray.y) / safeRs;
    }
    else
    {
        // In 2D_lensing.cpp, initial |v| = c so one affine step dLambda advances physical length c * dLambda.
        const double hDimensionless = (kSpeedOfLightMps * dLambdaAffine) / safeRs;
        const glm::dvec2 prevU = ray.uPos;
        Rk4StepCanonical2D(ray.uPos, ray.uVel, hDimensionless);

        ray.x = ray.uPos.x * safeRs;
        ray.y = ray.uPos.y * safeRs;
        ray.r = glm::length(ray.uPos) * safeRs;
        ray.phi = std::atan2(ray.y, ray.x);

        const double uR = std::max(glm::length(ray.uPos), 1e-9);
        const glm::dvec2 rHat = ray.uPos / uR;
        ray.dr = kSpeedOfLightMps * glm::dot(ray.uVel, rHat);
        ray.dphi = kSpeedOfLightMps * (ray.uPos.x * ray.uVel.y - ray.uPos.y * ray.uVel.x) /
                   (uR * uR * safeRs);

        // Check segment crossing with horizon |u| = 1
        double hitT = 1.0;
        if (SegmentHitsSphereD(glm::dvec3(prevU, 0.0), glm::dvec3(ray.uPos, 0.0), glm::dvec3(0.0),
                               1.0, hitT))
        {
            const glm::dvec2 horizonU = glm::mix(prevU, ray.uPos, hitT);
            ray.uPos = horizonU;
            ray.x = horizonU.x * safeRs;
            ray.y = horizonU.y * safeRs;
            ray.r = safeRs;
            ray.captured = true;
            return;
        }
    }

    if (ray.r <= safeRs)
    {
        ray.captured = true;
    }
    else if (ray.r > escapeRadiusMeters && ray.dr > 0.0)
    {
        ray.escaped = true;
    }
}

double EvaluatePlanarNullRelativeError(const PlanarRayState &ray, double rsMeters,
                                       PlanarGeodesicSolverKind solverKind)
{
    const double safeRs = std::max(rsMeters, 1.0);
    if (ray.r <= safeRs * 1.0001)
    {
        return 0.0;
    }

    if (solverKind == PlanarGeodesicSolverKind::CanonicalDimensionlessRk4)
    {
        const double uR = glm::length(ray.uPos);
        const double uL = ray.uPos.x * ray.uVel.y - ray.uPos.y * ray.uVel.x;
        const double instESq = glm::dot(ray.uVel, ray.uVel) - (uL * uL) / (uR * uR * uR);
        const double denom = std::max(std::fabs(ray.uE0Squared), 1e-12);
        return std::fabs(instESq - ray.uE0Squared) / denom;
    }

    const double f = 1.0 - safeRs / ray.r;
    const double instESq = ray.dr * ray.dr + f * ray.r * ray.r * ray.dphi * ray.dphi;
    const double refESq = ray.E * ray.E;
    return std::fabs(instESq - refESq) / std::max(refESq, 1e-12);
}

double EvaluatePlanarAngularMomentumRelativeError(const PlanarRayState &ray, double rsMeters,
                                                  PlanarGeodesicSolverKind solverKind)
{
    (void)rsMeters;
    if (solverKind == PlanarGeodesicSolverKind::CanonicalDimensionlessRk4)
    {
        const double instL = ray.uPos.x * ray.uVel.y - ray.uPos.y * ray.uVel.x;
        const double denom = std::max(std::fabs(ray.uL0), 1e-12);
        return std::fabs(instL - ray.uL0) / denom;
    }
    const double instL = ray.r * ray.r * ray.dphi;
    return std::fabs(instL - ray.L) / std::max(std::fabs(ray.L), 1e-12);
}

glm::dvec3 PixelToCameraRayDirection(int px, int py, int width, int height,
                                     const BlackHoleSceneConfig &scene)
{
    const double ndcX = 2.0 * ((static_cast<double>(px) + 0.5) / static_cast<double>(width)) - 1.0;
    const double ndcY = 2.0 * ((static_cast<double>(py) + 0.5) / static_cast<double>(height)) - 1.0;
    const glm::dvec3 raw = ndcX * scene.aspect * scene.tanHalfFov * scene.camRight +
                           ndcY * scene.tanHalfFov * scene.camUp + scene.camForward;
    return glm::normalize(raw);
}

GeodesicTraceResult TraceBlackHoleRayCPU(const glm::dvec3 &rayOriginMeters,
                                         const glm::dvec3 &rayDirNormalized,
                                         const BlackHoleSceneConfig &scene,
                                         Geodesic3DSolverKind solverKind)
{
    GeodesicTraceResult res;
    res.color = scene.backgroundColor;
    const double rs = std::max(scene.rsMeters, 1.0);
    const glm::dvec3 startRel = rayOriginMeters - scene.bhCenterMeters;
    res.finalRelPosMeters = startRel;
    res.finalDirection = rayDirNormalized;

    if (solverKind == Geodesic3DSolverKind::GeometricSilhouette)
    {
        // Straight-line sphere intersection against event horizon (CPU-geodesic.cpp:444-455)
        const double b = 2.0 * glm::dot(startRel, rayDirNormalized);
        const double c0 = glm::dot(startRel, startRel) - rs * rs;
        const double disc = b * b - 4.0 * c0;
        if (disc > 0.0)
        {
            const double s = std::sqrt(disc);
            const double t1 = (-b - s) * 0.5;
            const double t2 = (-b + s) * 0.5;
            if (t1 > 0.0 || t2 > 0.0)
            {
                res.outcome = GeodesicOutcome::HitHorizon;
                res.color = scene.redSilhouetteStyle ? glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)
                                                     : glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            }
        }
        return res;
    }

    if (solverKind == Geodesic3DSolverKind::CanonicalDimensionlessRk4)
    {
        const double invRs = 1.0 / rs;
        glm::dvec3 uPos = startRel * invRs;
        glm::dvec3 uVel = glm::normalize(rayDirNormalized);
        const double hStep = scene.dLambdaMeters * invRs;
        const double uEscape = scene.escapeRadiusMeters * invRs;
        const double uDiskR1 = scene.diskInnerMeters * invRs;
        const double uDiskR2 = scene.diskOuterMeters * invRs;

        const glm::dvec3 uL0Vec = glm::cross(uPos, uVel);
        const double uL0Mag = glm::length(uL0Vec);
        const double uR0 = std::max(glm::length(uPos), 1.000001);
        const double uE0Sq = glm::dot(uVel, uVel) - glm::dot(uL0Vec, uL0Vec) / (uR0 * uR0 * uR0);

        const std::size_t objCount =
            std::min(scene.objects.size(), kMaxBlackHoleSceneObjects);

        for (int i = 0; i < scene.maxSteps; ++i)
        {
            const double rBefore = glm::length(uPos);
            if (rBefore <= 1.0)
            {
                res.outcome = GeodesicOutcome::HitHorizon;
                res.stepsTaken = i;
                break;
            }

            const glm::dvec3 prevUPos = uPos;
            Rk4StepCanonical3D(uPos, uVel, hStep);
            const double rAfter = glm::length(uPos);
            res.stepsTaken = i + 1;

            double bestT = 2.0;
            int bestEvent = 0; // 1 = horizon, 2 = disk, 3 = object
            glm::dvec3 uDiskHit(0.0);
            glm::vec4 hitObjColor(0.0f);
            glm::dvec3 uHitObjCenter(0.0);
            glm::dvec3 uHitObjPoint(0.0);

            double tHorizon = 1.0;
            if (rAfter <= 1.0 || SegmentHitsSphereD(prevUPos, uPos, glm::dvec3(0.0), 1.0, tHorizon))
            {
                bestT = std::min(tHorizon, 1.0);
                bestEvent = 1;
            }

            if (scene.diskEnabled && (prevUPos.y * uPos.y <= 0.0) && (prevUPos.y != uPos.y))
            {
                const double tDisk = std::clamp(prevUPos.y / (prevUPos.y - uPos.y), 0.0, 1.0);
                const glm::dvec3 crossPt = glm::mix(prevUPos, uPos, tDisk);
                const double rDisk = std::hypot(crossPt.x, crossPt.z);
                if (rDisk >= uDiskR1 && rDisk <= uDiskR2 && tDisk < bestT)
                {
                    bestT = tDisk;
                    bestEvent = 2;
                    uDiskHit = crossPt;
                }
            }

            for (std::size_t objIdx = 0; objIdx < objCount; ++objIdx)
            {
                const glm::dvec3 objWorld(scene.objects[objIdx].posRadiusMeters.x,
                                          scene.objects[objIdx].posRadiusMeters.y,
                                          scene.objects[objIdx].posRadiusMeters.z);
                const glm::dvec3 uCenter = (objWorld - scene.bhCenterMeters) * invRs;
                const double uRadius =
                    static_cast<double>(scene.objects[objIdx].posRadiusMeters.w) * invRs;
                if (glm::length(uCenter) < 0.25 && uRadius <= 1.05)
                {
                    continue;
                }
                double tObj = 1.0;
                if (SegmentHitsSphereD(prevUPos, uPos, uCenter, uRadius, tObj) && tObj < bestT)
                {
                    bestT = tObj;
                    bestEvent = 3;
                    hitObjColor = scene.objects[objIdx].color;
                    uHitObjCenter = uCenter;
                    uHitObjPoint = glm::mix(prevUPos, uPos, tObj);
                }
            }

            if (bestEvent == 1)
            {
                res.outcome = GeodesicOutcome::HitHorizon;
                res.color = scene.redSilhouetteStyle ? glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)
                                                     : glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                break;
            }
            if (bestEvent == 2)
            {
                res.outcome = GeodesicOutcome::HitDisk;
                const double rDisk = std::hypot(uDiskHit.x, uDiskHit.z);
                const float rNorm = static_cast<float>(
                    std::clamp((rDisk - uDiskR1) / std::max(uDiskR2 - uDiskR1, 1e-4), 0.0, 1.0));
                const float ring = 0.85f + 0.15f * std::sin(rNorm * 18.8495559f);
                const glm::vec3 innerHot(1.0f, 0.92f, 0.65f);
                const glm::vec3 outerWarm(0.95f, 0.28f, 0.08f);
                res.color = glm::vec4(glm::mix(innerHot, outerWarm, rNorm) * ring, 1.0f);
                break;
            }
            if (bestEvent == 3)
            {
                res.outcome = GeodesicOutcome::HitObject;
                const glm::dvec3 N = glm::normalize(uHitObjPoint - uHitObjCenter);
                const glm::dvec3 uCam = startRel * invRs;
                const glm::dvec3 V = glm::normalize(uCam - uHitObjPoint);
                const float ambient = 0.14f;
                const float diff = static_cast<float>(std::max(glm::dot(N, V), 0.0));
                const float intensity = ambient + (1.0f - ambient) * diff;
                res.color = glm::vec4(glm::vec3(hitObjColor) * intensity, 1.0f);
                break;
            }

            if (rAfter > uEscape && glm::dot(uPos, uVel) > 0.0)
            {
                res.outcome = GeodesicOutcome::Escaped;
                break;
            }
        }

        res.finalRelPosMeters = uPos * rs;
        const double vLen = glm::length(uVel);
        res.finalDirection = vLen > 0.0 ? (uVel / vLen) : rayDirNormalized;

        const double uR = glm::length(uPos);
        if (uR > 1.001)
        {
            const glm::dvec3 uLVec = glm::cross(uPos, uVel);
            const double instESq = glm::dot(uVel, uVel) - glm::dot(uLVec, uLVec) / (uR * uR * uR);
            res.nullRelativeError =
                std::fabs(instESq - uE0Sq) / std::max(std::fabs(uE0Sq), 1e-12);
            if (uL0Mag > 1e-9)
            {
                res.angularMomentumRelativeError =
                    std::fabs(glm::length(uLVec) - uL0Mag) / uL0Mag;
            }
        }
        return res;
    }

    // Legacy spherical solvers (Euler or 4-stage RK4)
    LegacySphericalRay3D ray = InitLegacySphericalRay3D(startRel, rayDirNormalized, rs);
    glm::dvec3 prevRel(ray.x, ray.y, ray.z);
    const std::size_t objCount = std::min(scene.objects.size(), kMaxBlackHoleSceneObjects);

    for (int i = 0; i < scene.maxSteps; ++i)
    {
        if (ray.r <= rs)
        {
            res.outcome = GeodesicOutcome::HitHorizon;
            res.color = scene.redSilhouetteStyle ? glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)
                                                 : glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            res.stepsTaken = i;
            break;
        }

        if (solverKind == Geodesic3DSolverKind::LegacySphericalEuler)
        {
            StepLegacySphericalEuler3D(ray, scene.dLambdaMeters, rs);
        }
        else
        {
            StepLegacySphericalRk43D(ray, scene.dLambdaMeters, rs);
        }
        res.stepsTaken = i + 1;
        const glm::dvec3 newRel(ray.x, ray.y, ray.z);

        if (ray.r <= rs)
        {
            res.outcome = GeodesicOutcome::HitHorizon;
            res.color = scene.redSilhouetteStyle ? glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)
                                                 : glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            break;
        }

        if (scene.diskEnabled && (prevRel.y * newRel.y < 0.0))
        {
            const double denom = prevRel.y - newRel.y;
            const double t =
                std::fabs(denom) > 1e-12 ? std::clamp(prevRel.y / denom, 0.0, 1.0) : 0.5;
            const glm::dvec3 hitRel = glm::mix(prevRel, newRel, t);
            const double rDisk = std::hypot(hitRel.x, hitRel.z);
            if (rDisk >= scene.diskInnerMeters && rDisk <= scene.diskOuterMeters)
            {
                res.outcome = GeodesicOutcome::HitDisk;
                const float rNorm = static_cast<float>(
                    std::clamp(rDisk / std::max(scene.diskOuterMeters, 1.0), 0.0, 1.0));
                const glm::vec3 diskCol(1.0f, rNorm, 0.2f);
                res.color = glm::vec4(diskCol * (1.15f - 0.5f * rNorm), 1.0f);
                break;
            }
        }

        bool hitObj = false;
        const glm::dvec3 worldPos = scene.bhCenterMeters + newRel;
        for (std::size_t objIdx = 0; objIdx < objCount; ++objIdx)
        {
            const glm::dvec3 center(scene.objects[objIdx].posRadiusMeters.x,
                                    scene.objects[objIdx].posRadiusMeters.y,
                                    scene.objects[objIdx].posRadiusMeters.z);
            const double radius = static_cast<double>(scene.objects[objIdx].posRadiusMeters.w);
            if (glm::distance(center, scene.bhCenterMeters) < rs * 0.25 && radius <= rs * 1.05)
            {
                continue;
            }
            if (glm::distance(worldPos, center) <= radius)
            {
                res.outcome = GeodesicOutcome::HitObject;
                const glm::dvec3 N = glm::normalize(worldPos - center);
                const glm::dvec3 V = glm::normalize(rayOriginMeters - worldPos);
                const float ambient = 0.12f;
                const float diff = static_cast<float>(std::max(glm::dot(N, V), 0.0));
                const float intensity = ambient + (1.0f - ambient) * diff;
                res.color = glm::vec4(glm::vec3(scene.objects[objIdx].color) * intensity, 1.0f);
                hitObj = true;
                break;
            }
        }
        if (hitObj)
        {
            break;
        }

        prevRel = newRel;
        if (ray.r > scene.escapeRadiusMeters)
        {
            res.outcome = GeodesicOutcome::Escaped;
            break;
        }
    }

    res.finalRelPosMeters = glm::dvec3(ray.x, ray.y, ray.z);
    return res;
}

void PackBlackHoleUbos(const BlackHoleSceneConfig &scene, bool cameraMoving, int outW, int outH,
                       GpuCameraUboStd140 &outCam, GpuDiskUboStd140 &outDisk,
                       GpuObjectsUboStd140 &outObjects)
{
    if (scene.objects.size() > kMaxBlackHoleSceneObjects)
    {
        throw std::out_of_range("BlackHoleSceneConfig exceeds 16-object std140 UBO limit");
    }

    outCam = GpuCameraUboStd140{};
    outCam.camPos_tanHalfFov =
        glm::vec4(glm::vec3(scene.camPosMeters), static_cast<float>(scene.tanHalfFov));
    outCam.camRight_aspect =
        glm::vec4(glm::vec3(scene.camRight), static_cast<float>(scene.aspect));
    outCam.camUp_moving = glm::vec4(glm::vec3(scene.camUp), cameraMoving ? 1.0f : 0.0f);
    outCam.camForward_maxSteps =
        glm::vec4(glm::vec3(scene.camForward), static_cast<float>(scene.maxSteps));
    outCam.bhCenter_rs =
        glm::vec4(glm::vec3(scene.bhCenterMeters), static_cast<float>(scene.rsMeters));
    outCam.stepParams = glm::vec4(static_cast<float>(scene.dLambdaMeters),
                                  static_cast<float>(scene.escapeRadiusMeters),
                                  static_cast<float>(outW), static_cast<float>(outH));

    outDisk = GpuDiskUboStd140{};
    outDisk.diskParams =
        glm::vec4(static_cast<float>(scene.diskInnerMeters),
                  static_cast<float>(scene.diskOuterMeters), scene.diskEnabled ? 1.0f : 0.0f,
                  static_cast<float>(scene.diskThicknessMeters));

    outObjects = GpuObjectsUboStd140{};
    const std::size_t count = std::min(scene.objects.size(), kMaxBlackHoleSceneObjects);
    outObjects.counts = glm::ivec4(static_cast<int>(count), 0, 0, 0);
    for (std::size_t i = 0; i < count; ++i)
    {
        outObjects.objPosRadius[i] = scene.objects[i].posRadiusMeters;
        outObjects.objColor[i] = scene.objects[i].color;
        outObjects.objMassPad[i] = glm::vec4(scene.objects[i].massKg, 0.0f, 0.0f, 0.0f);
    }
}

// ============================================================================
// CpuRayWorker Implementation
// ============================================================================

CpuRayWorker::CpuRayWorker() = default;

CpuRayWorker::~CpuRayWorker() { StopAndJoin(); }

bool CpuRayWorker::IsOpenMpCompiled()
{
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

void CpuRayWorker::Start(std::uint32_t desiredThreadCount)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_)
    {
        return;
    }

    const unsigned int hw = std::max(std::thread::hardware_concurrency(), 1u);
    workerThreadCount_ = desiredThreadCount > 0 ? desiredThreadCount : std::max(1u, hw > 1 ? hw - 1 : 1u);
    cancelFlag_.store(false, std::memory_order_relaxed);
    running_ = true;
    coordinatorThread_ = std::thread(&CpuRayWorker::CoordinatorLoop, this);
}

void CpuRayWorker::SubmitJob(const CpuImageJobMeta &meta, CpuTileRenderFn renderFn, int tileSize)
{
    if (meta.width <= 0 || meta.height <= 0 || !renderFn)
    {
        return;
    }
    if (!running_)
    {
        Start();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Signal any currently executing job to abort at its next tile/row checkpoint
        cancelFlag_.store(true, std::memory_order_relaxed);
        pendingJob_.meta = meta;
        pendingJob_.renderFn = std::move(renderFn);
        pendingJob_.tileSize = std::max(tileSize, 8);
        hasPendingJob_ = true;
    }
    cv_.notify_one();
}

void CpuRayWorker::RequestCancel()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        hasPendingJob_ = false;
        hasCompletedResult_ = false;
        cancelFlag_.store(true, std::memory_order_relaxed);
    }
    cv_.notify_all();
}

bool CpuRayWorker::IsIdle() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return !hasPendingJob_ && !jobExecuting_;
}

float CpuRayWorker::Progress() const
{
    const int tot = std::max(totalTiles_.load(std::memory_order_relaxed), 1);
    const int done = completedTiles_.load(std::memory_order_relaxed);
    return std::clamp(static_cast<float>(done) / static_cast<float>(tot), 0.0f, 1.0f);
}

void CpuRayWorker::StopAndJoin()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_)
        {
            hasPendingJob_ = false;
            hasCompletedResult_ = false;
            std::vector<std::uint8_t>().swap(completedResult_.rgbaPixels);
            return;
        }
        running_ = false;
        hasPendingJob_ = false;
        cancelFlag_.store(true, std::memory_order_relaxed);
    }
    cv_.notify_all();
    if (coordinatorThread_.joinable())
    {
        coordinatorThread_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    hasCompletedResult_ = false;
    std::vector<std::uint8_t>().swap(completedResult_.rgbaPixels);
}

bool CpuRayWorker::TryPopCompletedResult(CpuImageResult &outResult,
                                         std::uint64_t expectedSceneGeneration,
                                         std::uint64_t expectedCameraRevision, int expectedWidth,
                                         int expectedHeight)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hasCompletedResult_)
    {
        return false;
    }

    CpuImageResult candidate = std::move(completedResult_);
    hasCompletedResult_ = false;

    if (candidate.meta.sceneGeneration != expectedSceneGeneration ||
        candidate.meta.cameraRevision != expectedCameraRevision ||
        candidate.meta.width != expectedWidth || candidate.meta.height != expectedHeight)
    {
        // Stale packet from prior mode/camera/resize: drop silently.
        return false;
    }

    outResult = std::move(candidate);
    return true;
}

void CpuRayWorker::CoordinatorLoop()
{
    for (;;)
    {
        PendingJob job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return !running_ || hasPendingJob_; });
            if (!running_)
            {
                return;
            }
            job = std::move(pendingJob_);
            hasPendingJob_ = false;
            jobExecuting_ = true;
            cancelFlag_.store(false, std::memory_order_relaxed);
        }

        const auto startWall = std::chrono::steady_clock::now();
        const int W = job.meta.width;
        const int H = job.meta.height;
        const int tile = job.tileSize;
        const int tilesX = (W + tile - 1) / tile;
        const int tilesY = (H + tile - 1) / tile;
        const int numTiles = std::max(tilesX * tilesY, 1);

        totalTiles_.store(numTiles, std::memory_order_relaxed);
        completedTiles_.store(0, std::memory_order_relaxed);

        std::vector<std::uint8_t> rgba(static_cast<std::size_t>(W) * static_cast<std::size_t>(H) *
                                           4u,
                                       255u);

        std::atomic<int> nextTileIndex{0};
        const std::uint32_t numWorkers = std::max(workerThreadCount_, 1u);

        auto workerFn = [&]()
        {
            for (;;)
            {
                if (cancelFlag_.load(std::memory_order_relaxed))
                {
                    break;
                }
                const int tileIdx = nextTileIndex.fetch_add(1, std::memory_order_relaxed);
                if (tileIdx >= numTiles)
                {
                    break;
                }
                const int ty = tileIdx / tilesX;
                const int tx = tileIdx % tilesX;
                const int x0 = tx * tile;
                const int y0 = ty * tile;
                const int x1 = std::min(x0 + tile, W);
                const int y1 = std::min(y0 + tile, H);

                job.renderFn(x0, y0, x1, y1, W, H, rgba.data(), cancelFlag_);
                completedTiles_.fetch_add(1, std::memory_order_relaxed);
            }
        };

        std::vector<std::thread> pool;
        if (numWorkers > 1 && numTiles > 1)
        {
            pool.reserve(numWorkers - 1);
            for (std::uint32_t t = 1; t < numWorkers; ++t)
            {
                pool.emplace_back(workerFn);
            }
        }
        workerFn();
        for (auto &th : pool)
        {
            if (th.joinable())
            {
                th.join();
            }
        }

        const double elapsedMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startWall)
                .count();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobExecuting_ = false;
            if (!cancelFlag_.load(std::memory_order_relaxed) && running_ && !hasPendingJob_)
            {
                completedResult_.meta = job.meta;
                completedResult_.rgbaPixels = std::move(rgba);
                completedResult_.elapsedMs = elapsedMs;
                completedResult_.workerThreadsUsed = numWorkers;
                completedResult_.usedOpenMp = IsOpenMpCompiled();
                hasCompletedResult_ = true;
            }
        }
    }
}

} // namespace grav_models

