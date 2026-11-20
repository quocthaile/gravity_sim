#include "error_utils.h"

#include <cmath>
#include <stdexcept>

namespace
{
void ValidateFiniteVector(const glm::dvec3 &vector)
{
    if (!std::isfinite(vector.x) || !std::isfinite(vector.y) || !std::isfinite(vector.z))
    {
        throw std::invalid_argument("Error metric input contains a non-finite vector");
    }
}

void ValidateMatchingSamples(const std::vector<glm::dvec3> &actual,
                             const std::vector<glm::dvec3> &reference)
{
    if (actual.size() != reference.size())
    {
        throw std::invalid_argument("Error metric inputs have different sample counts");
    }
    if (actual.empty())
    {
        throw std::invalid_argument("Error metrics require at least one sample");
    }
}
} // namespace

double VectorMagnitude(const glm::dvec3 &vector)
{
    ValidateFiniteVector(vector);
    const double magnitude = std::hypot(vector.x, vector.y, vector.z);
    if (!std::isfinite(magnitude))
    {
        throw std::overflow_error("Vector magnitude is not finite");
    }
    return magnitude;
}

double VectorError(const glm::dvec3 &actual, const glm::dvec3 &reference)
{
    ValidateFiniteVector(actual);
    ValidateFiniteVector(reference);
    return VectorMagnitude(actual - reference);
}

double CalculateVectorRMSE(const std::vector<glm::dvec3> &actual,
                           const std::vector<glm::dvec3> &reference)
{
    ValidateMatchingSamples(actual, reference);
    long double sumSquaredError = 0.0L;
    for (std::size_t index = 0; index < actual.size(); ++index)
    {
        const double error = VectorError(actual[index], reference[index]);
        sumSquaredError += static_cast<long double>(error) * error;
    }

    const double rmse = std::sqrt(static_cast<double>(sumSquaredError / actual.size()));
    if (!std::isfinite(rmse))
    {
        throw std::overflow_error("Vector RMSE is not finite");
    }
    return rmse;
}

MagnitudeMAPEResult CalculateMagnitudeMAPE(const std::vector<glm::dvec3> &actual,
                                          const std::vector<glm::dvec3> &reference,
                                          double referenceMagnitudeThreshold,
                                          VectorMagnitudeFunction magnitudeFunction)
{
    ValidateMatchingSamples(actual, reference);
    if (!std::isfinite(referenceMagnitudeThreshold) || referenceMagnitudeThreshold < 0.0)
    {
        throw std::invalid_argument("MAPE reference threshold must be finite and non-negative");
    }
    if (magnitudeFunction == nullptr)
    {
        throw std::invalid_argument("MAPE magnitude function must not be null");
    }

    long double percentageSum = 0.0L;
    MagnitudeMAPEResult result;
    for (std::size_t index = 0; index < actual.size(); ++index)
    {
        ValidateFiniteVector(actual[index]);
        ValidateFiniteVector(reference[index]);
        const double actualMagnitude = magnitudeFunction(actual[index]);
        const double referenceMagnitude = magnitudeFunction(reference[index]);
        if (!std::isfinite(actualMagnitude) || !std::isfinite(referenceMagnitude) ||
            actualMagnitude < 0.0 || referenceMagnitude < 0.0)
        {
            throw std::invalid_argument("MAPE magnitude function returned an invalid value");
        }

        if (referenceMagnitude <= referenceMagnitudeThreshold)
        {
            ++result.excludedSamples;
            continue;
        }

        percentageSum +=
            std::abs(actualMagnitude - referenceMagnitude) / referenceMagnitude * 100.0;
        ++result.includedSamples;
    }

    if (result.includedSamples != 0)
    {
        const double percentage =
            static_cast<double>(percentageSum / result.includedSamples);
        if (!std::isfinite(percentage))
        {
            throw std::overflow_error("Magnitude MAPE is not finite");
        }
        result.percentage = percentage;
    }
    return result;
}
