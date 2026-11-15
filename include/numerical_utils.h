#pragma once

#include <vector>

namespace numerical
{

struct ScalarPair
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

struct VectorPair
{
    Vector3 reference;
    Vector3 actual;
};

using MetricFunction = double (*)(const std::vector<ScalarPair> &, double);

struct ErrorMetrics
{
    double rmse;
    double mape;
    double maxAbsoluteError;
    double relativeError;
};

struct VectorErrorMetrics
{
    ErrorMetrics componentMetrics;
    double positionRmse;
    double maxPositionError;
};

double CalculateRmse(const std::vector<ScalarPair> &samples, double zeroTolerance);
double CalculateMape(const std::vector<ScalarPair> &samples, double zeroTolerance);
double CalculateMaxAbsoluteError(const std::vector<ScalarPair> &samples, double zeroTolerance);
double CalculateRelativeError(const std::vector<ScalarPair> &samples, double zeroTolerance);
ErrorMetrics CalculateMetrics(const std::vector<ScalarPair> &samples, double zeroTolerance);
VectorErrorMetrics CalculateVectorMetrics(const std::vector<VectorPair> &samples,
                                          double zeroTolerance);

} // namespace numerical
