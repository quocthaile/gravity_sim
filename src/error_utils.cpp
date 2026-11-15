#include "error_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace error_utils
{
namespace
{
constexpr std::array<MetricFunction, 4> kMetricFunctions = {
    CalculateRmse,
    CalculateMape,
    CalculateMaxAbsoluteError,
    CalculateRelativeError,
};

void ValidateInput(const std::vector<ScalarSample> &samples, double zeroTolerance)
{
    if (!std::isfinite(zeroTolerance) || zeroTolerance < 0.0)
    {
        throw std::invalid_argument("zeroTolerance must be finite and non-negative");
    }
    for (const ScalarSample &sample : samples)
    {
        if (!std::isfinite(sample.reference) || !std::isfinite(sample.actual))
        {
            throw std::invalid_argument("error metric samples must be finite");
        }
    }
}

double AbsoluteError(const ScalarSample &sample)
{
    return std::abs(sample.actual - sample.reference);
}

double VectorLength(const Vector3 &value)
{
    return std::hypot(value.x, value.y, value.z);
}

MetricResult MakeResult(double value, std::size_t sampleCount, std::size_t validSampleCount,
                        std::size_t skippedReferenceCount)
{
    return {value, sampleCount, validSampleCount, skippedReferenceCount, validSampleCount != 0};
}
} // namespace

MetricResult CalculateRmse(const std::vector<ScalarSample> &samples, double zeroTolerance)
{
    ValidateInput(samples, zeroTolerance);
    double sumSquaredError = 0.0;
    for (const ScalarSample &sample : samples)
    {
        const double error = AbsoluteError(sample);
        sumSquaredError += error * error;
    }
    if (samples.empty())
    {
        return {};
    }
    return MakeResult(std::sqrt(sumSquaredError / static_cast<double>(samples.size())),
                      samples.size(), samples.size(), 0);
}

MetricResult CalculateMape(const std::vector<ScalarSample> &samples, double zeroTolerance)
{
    ValidateInput(samples, zeroTolerance);
    double sumPercentageError = 0.0;
    std::size_t validSampleCount = 0;
    for (const ScalarSample &sample : samples)
    {
        const double referenceMagnitude = std::abs(sample.reference);
        if (referenceMagnitude <= zeroTolerance)
        {
            continue;
        }
        sumPercentageError += AbsoluteError(sample) / referenceMagnitude;
        ++validSampleCount;
    }
    const std::size_t skippedCount = samples.size() - validSampleCount;
    if (validSampleCount == 0)
    {
        return MakeResult(0.0, samples.size(), 0, skippedCount);
    }
    return MakeResult(100.0 * sumPercentageError / static_cast<double>(validSampleCount),
                      samples.size(), validSampleCount, skippedCount);
}

MetricResult CalculateMaxAbsoluteError(const std::vector<ScalarSample> &samples,
                                       double zeroTolerance)
{
    ValidateInput(samples, zeroTolerance);
    double maximumError = 0.0;
    for (const ScalarSample &sample : samples)
    {
        maximumError = std::max(maximumError, AbsoluteError(sample));
    }
    if (samples.empty())
    {
        return {};
    }
    return MakeResult(maximumError, samples.size(), samples.size(), 0);
}

MetricResult CalculateRelativeError(const std::vector<ScalarSample> &samples,
                                    double zeroTolerance)
{
    ValidateInput(samples, zeroTolerance);
    double sumAbsoluteError = 0.0;
    double sumReferenceMagnitude = 0.0;
    std::size_t validSampleCount = 0;
    for (const ScalarSample &sample : samples)
    {
        const double referenceMagnitude = std::abs(sample.reference);
        if (referenceMagnitude <= zeroTolerance)
        {
            continue;
        }
        sumAbsoluteError += AbsoluteError(sample);
        sumReferenceMagnitude += referenceMagnitude;
        ++validSampleCount;
    }
    const std::size_t skippedCount = samples.size() - validSampleCount;
    if (validSampleCount == 0)
    {
        return MakeResult(0.0, samples.size(), 0, skippedCount);
    }
    return MakeResult(sumAbsoluteError / sumReferenceMagnitude, samples.size(),
                      validSampleCount, skippedCount);
}

ErrorMetrics CalculateMetrics(const std::vector<ScalarSample> &samples, double zeroTolerance)
{
    ErrorMetrics metrics;
    MetricResult *results[] = {&metrics.rmse, &metrics.mape, &metrics.maxAbsoluteError,
                               &metrics.relativeError};
    for (std::size_t index = 0; index < kMetricFunctions.size(); ++index)
    {
        *results[index] = kMetricFunctions[index](samples, zeroTolerance);
    }
    return metrics;
}

VectorErrorMetrics CalculateVectorMetrics(const std::vector<VectorSample> &samples,
                                          double zeroTolerance)
{
    if (!std::isfinite(zeroTolerance) || zeroTolerance < 0.0)
    {
        throw std::invalid_argument("zeroTolerance must be finite and non-negative");
    }

    std::vector<ScalarSample> magnitudeSamples;
    magnitudeSamples.reserve(samples.size());
    double sumSquaredVectorError = 0.0;
    double maximumVectorError = 0.0;
    for (const VectorSample &sample : samples)
    {
        const double referenceMagnitude = VectorLength(sample.reference);
        const double actualMagnitude = VectorLength(sample.actual);
        const double dx = sample.actual.x - sample.reference.x;
        const double dy = sample.actual.y - sample.reference.y;
        const double dz = sample.actual.z - sample.reference.z;
        const double vectorError = VectorLength({dx, dy, dz});
        if (!std::isfinite(referenceMagnitude) || !std::isfinite(actualMagnitude) ||
            !std::isfinite(vectorError))
        {
            throw std::invalid_argument("error metric vector samples must be finite");
        }
        magnitudeSamples.push_back({referenceMagnitude, referenceMagnitude + vectorError});
        sumSquaredVectorError += vectorError * vectorError;
        maximumVectorError = std::max(maximumVectorError, vectorError);
    }

    MetricResult vectorRmse;
    MetricResult maximumVectorErrorResult;
    if (!samples.empty())
    {
        vectorRmse = MakeResult(
            std::sqrt(sumSquaredVectorError / static_cast<double>(samples.size())), samples.size(),
            samples.size(), 0);
        maximumVectorErrorResult =
            MakeResult(maximumVectorError, samples.size(), samples.size(), 0);
    }
    return {CalculateMetrics(magnitudeSamples, zeroTolerance), vectorRmse,
            maximumVectorErrorResult};
}
} // namespace error_utils
