#pragma once

#include <cstddef>
#include <vector>

namespace error_utils
{

struct ScalarSample
{
    double reference;
    double actual;
};

struct Vector3
{
    double x;
    double y;
    double z;
};

struct VectorSample
{
    Vector3 reference;
    Vector3 actual;
};

struct MetricResult
{
    double value = 0.0;
    std::size_t sampleCount = 0;
    std::size_t validSampleCount = 0;
    std::size_t skippedReferenceCount = 0;
    bool available = false;
};

using MetricFunction = MetricResult (*)(const std::vector<ScalarSample> &, double);

struct ErrorMetrics
{
    MetricResult rmse;
    MetricResult mape;
    MetricResult maxAbsoluteError;
    MetricResult relativeError;
};

struct VectorErrorMetrics
{
    ErrorMetrics magnitudeMetrics;
    MetricResult vectorRmse;
    MetricResult maxVectorError;
};

MetricResult CalculateRmse(const std::vector<ScalarSample> &samples, double zeroTolerance);
MetricResult CalculateMape(const std::vector<ScalarSample> &samples, double zeroTolerance);
MetricResult CalculateMaxAbsoluteError(const std::vector<ScalarSample> &samples,
                                       double zeroTolerance);
MetricResult CalculateRelativeError(const std::vector<ScalarSample> &samples,
                                    double zeroTolerance);
ErrorMetrics CalculateMetrics(const std::vector<ScalarSample> &samples, double zeroTolerance);
VectorErrorMetrics CalculateVectorMetrics(const std::vector<VectorSample> &samples,
                                          double zeroTolerance);

} // namespace error_utils
