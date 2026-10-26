#pragma once

// CPU-only test support: also usable on the Mac without CUDA headers.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace softmax_test {

// Baseline FP32 acceptance criteria, fixed before GPU testing.
// Elementwise: |actual - reference| <= atol + rtol * |reference|.
constexpr double atol = 1e-7;
constexpr double rtol = 1e-5;
constexpr double row_sum_atol = 1e-5;

enum class Pattern { Mixed, Equal, Negative, LargePositive, LargeNegative, Tied, Peaked };

struct Case {
    const char* name;
    int rows;
    int cols;
    Pattern pattern;
};

inline std::vector<Case> cases() {
    return {
        {"mixed_initial", 37, 130, Pattern::Mixed},
        {"single_column", 3, 1, Pattern::Mixed},
        {"equal_short", 3, 17, Pattern::Equal},
        {"negative_127", 4, 127, Pattern::Negative},
        {"equal_128", 3, 128, Pattern::Equal},
        {"large_positive_129", 3, 129, Pattern::LargePositive},
        {"large_negative_130", 3, 130, Pattern::LargeNegative},
        {"tied_maxima_257", 3, 257, Pattern::Tied},
        {"dominant_tail_1025", 3, 1025, Pattern::Peaked},
        {"mixed_long_4097", 2, 4097, Pattern::Mixed},
    };
}

inline std::vector<float> make_input(const Case& test) {
    std::vector<float> x(test.rows * test.cols);
    for (int row = 0; row < test.rows; ++row) {
        for (int col = 0; col < test.cols; ++col) {
            const float base = static_cast<float>((col * 17 + row * 13) % 41 - 20) * 0.25f;
            float value = base;
            switch (test.pattern) {
                case Pattern::Mixed: break;
                case Pattern::Equal: value = static_cast<float>(row - 1); break;
                case Pattern::Negative: value = -10.0f + base; break;
                case Pattern::LargePositive: value = 1000.0f + base; break;
                case Pattern::LargeNegative: value = -1000.0f + base; break;
                case Pattern::Tied:
                    value = (col == row % test.cols || col == test.cols - 1) ? 8.0f : -8.0f;
                    break;
                case Pattern::Peaked:
                    value = col == test.cols - 1 - row ? 1000.0f : -1000.0f;
                    break;
            }
            x[row * test.cols + col] = value;
        }
    }
    return x;
}

inline std::vector<double> reference(const std::vector<float>& x, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || x.size() != static_cast<size_t>(rows) * cols) {
        throw std::invalid_argument("invalid softmax reference shape");
    }
    std::vector<double> result(x.size());
    for (int row = 0; row < rows; ++row) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < cols; ++col) {
            const double value = x[row * cols + col];
            if (!std::isfinite(value)) throw std::invalid_argument("reference expects finite inputs");
            maximum = std::max(maximum, value);
        }
        double denominator = 0.0;
        for (int col = 0; col < cols; ++col) {
            const int index = row * cols + col;
            result[index] = std::exp(static_cast<double>(x[index]) - maximum);
            denominator += result[index];
        }
        for (int col = 0; col < cols; ++col) result[row * cols + col] /= denominator;
    }
    return result;
}

struct Result {
    int bad_elements = 0;
    int bad_rows = 0;
    double max_abs_error = 0.0;
    double max_row_sum_error = 0.0;
    bool passed() const { return bad_elements == 0 && bad_rows == 0; }
};

inline Result validate(const std::vector<float>& y, const std::vector<double>& expected,
                       int rows, int cols) {
    if (rows <= 0 || cols <= 0 || y.size() != static_cast<size_t>(rows) * cols ||
        expected.size() != y.size()) throw std::invalid_argument("invalid validation shape");
    Result result;
    for (int row = 0; row < rows; ++row) {
        double sum = 0.0;
        bool finite_row = true;
        for (int col = 0; col < cols; ++col) {
            const int index = row * cols + col;
            const double actual = y[index];
            const double error = std::abs(actual - expected[index]);
            const bool finite = std::isfinite(actual) && std::isfinite(expected[index]);
            if (!finite || actual < 0.0 || actual > 1.0 ||
                error > atol + rtol * std::abs(expected[index])) ++result.bad_elements;
            result.max_abs_error = std::max(result.max_abs_error,
                finite ? error : std::numeric_limits<double>::infinity());
            finite_row = finite_row && finite;
            sum += actual;
        }
        const double error = finite_row ? std::abs(sum - 1.0)
                                        : std::numeric_limits<double>::infinity();
        result.max_row_sum_error = std::max(result.max_row_sum_error, error);
        if (error > row_sum_atol) ++result.bad_rows;
    }
    return result;
}

} // namespace softmax_test
