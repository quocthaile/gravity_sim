#include "error_utils.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void RequireNear(double actual, double expected, double tolerance, const char *name)
{
    if (std::abs(actual - expected) > tolerance)
    {
        throw std::runtime_error(std::string(name) + " is outside tolerance");
    }
}
} // namespace

int main()
{
    const std::vector<error_utils::ScalarSample> scalarSamples = {{2.0, 3.0}, {0.0, 0.0}};
    const error_utils::ErrorMetrics scalarMetrics =
        error_utils::CalculateMetrics(scalarSamples, 1.0e-12);
    RequireNear(scalarMetrics.rmse.value, std::sqrt(0.5), 1.0e-12, "RMSE");
    RequireNear(scalarMetrics.mape.value, 50.0, 1.0e-12, "MAPE");
    RequireNear(scalarMetrics.maxAbsoluteError.value, 1.0, 1.0e-12, "maximum absolute error");
    RequireNear(scalarMetrics.relativeError.value, 0.5, 1.0e-12, "relative error");
    if (scalarMetrics.mape.skippedReferenceCount != 1)
    {
        throw std::runtime_error("MAPE did not report the zero reference sample");
    }

    const std::vector<error_utils::VectorSample> vectorSamples = {
        {{1.0, 2.0, 3.0}, {2.0, 2.0, 3.0}},
        {{0.0, 0.0, 0.0}, {0.0, 3.0, 4.0}},
    };
    const error_utils::VectorErrorMetrics vectorMetrics =
        error_utils::CalculateVectorMetrics(vectorSamples, 1.0e-12);
    RequireNear(vectorMetrics.vectorRmse.value, std::sqrt(13.0), 1.0e-12, "vector RMSE");
    RequireNear(vectorMetrics.maxVectorError.value, 5.0, 1.0e-12, "maximum vector error");
    RequireNear(vectorMetrics.magnitudeMetrics.mape.value, 100.0 / std::sqrt(14.0), 1.0e-12,
                "vector MAPE");
    if (vectorMetrics.magnitudeMetrics.mape.skippedReferenceCount != 1)
    {
        throw std::runtime_error("vector MAPE did not report a near-zero reference");
    }

    std::cout << "error_utils tests passed\n";
    return 0;
}
