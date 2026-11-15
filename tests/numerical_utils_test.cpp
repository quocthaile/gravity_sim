#include "numerical_utils.h"

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
    const std::vector<numerical::ScalarPair> scalarSamples = {{2.0, 3.0}, {0.0, 0.0}};
    const numerical::ErrorMetrics metrics = numerical::CalculateMetrics(scalarSamples, 1.0e-12);
    RequireNear(metrics.rmse, std::sqrt(0.5), 1.0e-12, "RMSE");
    RequireNear(metrics.mape, 50.0, 1.0e-12, "MAPE zero-reference handling");
    RequireNear(metrics.maxAbsoluteError, 1.0, 1.0e-12, "maximum absolute error");
    RequireNear(metrics.relativeError, 0.5, 1.0e-12, "relative error");

    const std::vector<numerical::VectorPair> vectorSamples = {
        {{1.0, 2.0, 3.0}, {2.0, 2.0, 3.0}},
        {{0.0, 0.0, 0.0}, {0.0, 3.0, 4.0}},
    };
    const numerical::VectorErrorMetrics vectorMetrics =
        numerical::CalculateVectorMetrics(vectorSamples, 1.0e-12);
    RequireNear(vectorMetrics.positionRmse, std::sqrt(13.0), 1.0e-12, "position RMSE");
    RequireNear(vectorMetrics.maxPositionError, 5.0, 1.0e-12, "maximum position error");

    std::cout << "numerical_utils tests passed\n";
    return 0;
}
