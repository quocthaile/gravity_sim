#include "numerical_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace numerical
{
namespace
{
constexpr std::array<MetricFunction, 4> kMetricFunctions = {
    CalculateRmse,
    CalculateMape,
    CalculateMaxAbsoluteError,
    CalculateRelativeError,
};

double AbsoluteError(const ScalarPair &sample)
{
    return std::abs(sample.actual - sample.reference);
}

double VectorLength(const Vector3 &value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

} // namespace

double CalculateRmse(const std::vector<ScalarPair> &samples, double zeroTolerance)
{
    (void)zeroTolerance;
    if (samples.empty())
    {
        return 0.0;
    }

    double sumSquaredError = 0.0;
    for (const ScalarPair &sample : samples)
    {
        const double error = AbsoluteError(sample);
        sumSquaredError += error * error;
    }
    return std::sqrt(sumSquaredError / static_cast<double>(samples.size()));
}

double CalculateMape(const std::vector<ScalarPair> &samples, double zeroTolerance)
{
    if (zeroTolerance < 0.0)
    {
        throw std::invalid_argument("zeroTolerance must be non-negative");
    }
    double sumPercentageError = 0.0;
    std::size_t validSampleCount = 0;
    for (const ScalarPair &sample : samples)
    {
        if (std::abs(sample.reference) <= zeroTolerance)
        {
            continue;
        }
        sumPercentageError += AbsoluteError(sample) / std::abs(sample.reference);
        ++validSampleCount;
    }
    return validSampleCount == 0
               ? 0.0
               : 100.0 * sumPercentageError / static_cast<double>(validSampleCount);
}

double CalculateMaxAbsoluteError(const std::vector<ScalarPair> &samples, double zeroTolerance)
{
    (void)zeroTolerance;
    double maximumError = 0.0;
    for (const ScalarPair &sample : samples)
    {
        maximumError = std::max(maximumError, AbsoluteError(sample));
    }
    return maximumError;
}

double CalculateRelativeError(const std::vector<ScalarPair> &samples, double zeroTolerance)
{
    if (zeroTolerance < 0.0)
    {
        throw std::invalid_argument("zeroTolerance must be non-negative");
    }
    double sumAbsoluteError = 0.0;
    double sumReferenceMagnitude = 0.0;
    for (const ScalarPair &sample : samples)
    {
        sumAbsoluteError += AbsoluteError(sample);
        sumReferenceMagnitude += std::abs(sample.reference);
    }
    if (sumReferenceMagnitude <= zeroTolerance)
    {
        if (sumAbsoluteError <= zeroTolerance)
        {
            return 0.0;
        }
        return sumAbsoluteError / std::max(zeroTolerance, std::numeric_limits<double>::min());
    }
    return sumAbsoluteError / sumReferenceMagnitude;
}

ErrorMetrics CalculateMetrics(const std::vector<ScalarPair> &samples, double zeroTolerance)
{
    return {kMetricFunctions[0](samples, zeroTolerance),
            kMetricFunctions[1](samples, zeroTolerance),
            kMetricFunctions[2](samples, zeroTolerance),
            kMetricFunctions[3](samples, zeroTolerance)};
}

VectorErrorMetrics CalculateVectorMetrics(const std::vector<VectorPair> &samples,
                                          double zeroTolerance)
{
    std::vector<ScalarPair> componentSamples;
    componentSamples.reserve(samples.size() * 3);
    double sumSquaredPositionError = 0.0;
    double maximumPositionError = 0.0;
    for (const VectorPair &sample : samples)
    {
        componentSamples.push_back({sample.reference.x, sample.actual.x});
        componentSamples.push_back({sample.reference.y, sample.actual.y});
        componentSamples.push_back({sample.reference.z, sample.actual.z});
        const Vector3 difference = {sample.actual.x - sample.reference.x,
                                    sample.actual.y - sample.reference.y,
                                    sample.actual.z - sample.reference.z};
        const double positionError = VectorLength(difference);
        sumSquaredPositionError += positionError * positionError;
        maximumPositionError = std::max(maximumPositionError, positionError);
    }

    const ErrorMetrics metrics = CalculateMetrics(componentSamples, zeroTolerance);
    const double positionRmse =
        samples.empty() ? 0.0 : std::sqrt(sumSquaredPositionError / samples.size());
    return {metrics, positionRmse, maximumPositionError};
}

} // namespace numerical
