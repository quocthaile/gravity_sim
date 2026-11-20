#include "error_utils.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
void Check(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void TestVectorErrorAndRMSE()
{
    const glm::dvec3 zero(0.0);
    const glm::dvec3 error(3.0, 4.0, 0.0);
    Check(std::abs(VectorError(error, zero) - 5.0) < 1.0e-12,
          "Vector error should be Euclidean distance");
    Check(std::abs(CalculateVectorRMSE({error, zero}, {zero, zero}) - std::sqrt(12.5)) <
              1.0e-12,
          "Vector RMSE should average squared vector errors");
}

void TestMagnitudeMAPEExcludesSmallReferences()
{
    const MagnitudeMAPEResult result =
        CalculateMagnitudeMAPE({glm::dvec3(2.0, 0.0, 0.0), glm::dvec3(5.0, 0.0, 0.0)},
                               {glm::dvec3(1.0, 0.0, 0.0), glm::dvec3(0.0)},
                               1.0e-9, &VectorMagnitude);
    Check(result.percentage.has_value(), "MAPE should be available with a valid reference");
    Check(std::abs(*result.percentage - 100.0) < 1.0e-12,
          "MAPE should compare vector magnitudes");
    Check(result.includedSamples == 1 && result.excludedSamples == 1,
          "MAPE should report included and excluded samples");
}

void TestMagnitudeMAPECanBeUnavailable()
{
    const MagnitudeMAPEResult result =
        CalculateMagnitudeMAPE({glm::dvec3(0.0)}, {glm::dvec3(0.0)}, 1.0e-9, &VectorMagnitude);
    Check(!result.percentage.has_value(), "MAPE must be unavailable when all references are small");
    Check(result.includedSamples == 0 && result.excludedSamples == 1,
          "Unavailable MAPE should retain its sample counts");
}

void TestInvalidInputsAreRejected()
{
    bool mismatchedSamplesRejected = false;
    try
    {
        CalculateVectorRMSE({glm::dvec3(0.0)}, {});
    }
    catch (const std::invalid_argument &)
    {
        mismatchedSamplesRejected = true;
    }
    Check(mismatchedSamplesRejected, "Mismatched vector samples must be rejected");

    bool nonFiniteRejected = false;
    try
    {
        VectorMagnitude(glm::dvec3(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0));
    }
    catch (const std::invalid_argument &)
    {
        nonFiniteRejected = true;
    }
    Check(nonFiniteRejected, "Non-finite vector inputs must be rejected");
}
} // namespace

int main()
{
    try
    {
        TestVectorErrorAndRMSE();
        TestMagnitudeMAPEExcludesSmallReferences();
        TestMagnitudeMAPECanBeUnavailable();
        TestInvalidInputsAreRejected();
        std::cout << "error_utils_test: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error_utils_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
