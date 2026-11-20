#pragma once

#include <glm/glm.hpp>
#include <cstddef>
#include <optional>
#include <vector>

using VectorMagnitudeFunction = double (*)(const glm::dvec3 &);

struct MagnitudeMAPEResult
{
    std::optional<double> percentage;
    std::size_t includedSamples = 0;
    std::size_t excludedSamples = 0;
};

double VectorMagnitude(const glm::dvec3 &vector);
double VectorError(const glm::dvec3 &actual, const glm::dvec3 &reference);
double CalculateVectorRMSE(const std::vector<glm::dvec3> &actual,
                           const std::vector<glm::dvec3> &reference);
MagnitudeMAPEResult CalculateMagnitudeMAPE(const std::vector<glm::dvec3> &actual,
                                          const std::vector<glm::dvec3> &reference,
                                          double referenceMagnitudeThreshold,
                                          VectorMagnitudeFunction magnitudeFunction);
