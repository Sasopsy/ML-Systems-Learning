// CPU-only checks of the reference and validator; this does NOT run the GPU kernel.
#include "softmax_test_support.h"
#include <cstdio>
#include <cstdlib>

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    using namespace softmax_test;
    const auto uniform = reference({-1000.0f, -1000.0f, -1000.0f}, 1, 3);
    for (double value : uniform) require(std::abs(value - 1.0 / 3.0) < 1e-14, "uniform reference");
    require(reference({1000.0f}, 1, 1)[0] == 1.0, "single-column reference");
    const auto two = reference({0.0f, 1.0f}, 1, 2);
    require(std::abs(two[0] - 1.0 / (1.0 + std::exp(1.0))) < 1e-14, "analytic two-logit reference");
    const auto shifted = reference({1000.0f, 1001.0f}, 1, 2);
    require(std::abs(two[0] - shifted[0]) < 1e-14, "shift invariance");

    for (const auto& test : cases()) {
        const auto expected = reference(make_input(test), test.rows, test.cols);
        const std::vector<float> rounded(expected.begin(), expected.end());
        require(validate(rounded, expected, test.rows, test.cols).passed(), "rounded reference acceptance");
    }

    const std::vector<double> expected{0.5, 0.5};
    require(!validate({0.0f, 0.0f}, expected, 1, 2).passed(), "reject zero output");
    require(!validate({-0.1f, 1.1f}, expected, 1, 2).passed(), "reject invalid probabilities");
    require(!validate({std::numeric_limits<float>::quiet_NaN(), 0.5f}, expected, 1, 2).passed(),
            "reject NaN output");
    require(!validate({std::numeric_limits<float>::infinity(), 0.5f}, expected, 1, 2).passed(),
            "reject infinite output");
    const auto nonuniform = reference({0.0f, 1.0f, 2.0f}, 1, 3);
    const std::vector<float> reversed{static_cast<float>(nonuniform[2]),
                                    static_cast<float>(nonuniform[1]),
                                    static_cast<float>(nonuniform[0])};
    require(!validate(reversed, nonuniform, 1, 3).passed(), "reject wrong output even if normalized");
    std::puts("PASS: CPU reference and validator checks; CUDA kernel was not executed.");
    return EXIT_SUCCESS;
}
