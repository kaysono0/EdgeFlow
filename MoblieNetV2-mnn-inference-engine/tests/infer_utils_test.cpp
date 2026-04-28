#include "infer_utils.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

bool near(float lhs, float rhs, float eps = 1.0e-5f) {
    return std::fabs(lhs - rhs) < eps;
}

}  // namespace

int main() {
    const std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    const std::vector<float> probs = softmax(logits);

    assert(probs.size() == logits.size());
    assert(near(probs[0] + probs[1] + probs[2], 1.0f));
    assert(probs[2] > probs[1]);
    assert(probs[1] > probs[0]);

    const std::vector<TopKItem> top2 = topk(probs, 2);
    assert(top2.size() == 2);
    assert(top2[0].class_id == 2);
    assert(top2[1].class_id == 1);

    const std::vector<double> times = {4.0, 1.0, 3.0, 2.0, 10.0};
    assert(near(static_cast<float>(mean_ms(times)), 4.0f));
    assert(near(static_cast<float>(median_ms(times)), 3.0f));
    assert(near(static_cast<float>(percentile_ms(times, 0.95)), 4.0f));

    return 0;
}
