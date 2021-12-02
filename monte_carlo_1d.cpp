#include <algorithm>
#include <execution>
#include <functional>
using namespace std::placeholders;
#include <boost/timer/timer.hpp>
using boost::timer::auto_cpu_timer;
#include <fmt/format.h>
#include <iostream>
#include <random>

double monte_carlo_integral(std::function<double(double)> f, double a, double b) {
  std::random_device               rd;
  std::mt19937                     gen(rd());
  std::uniform_real_distribution<> dis(a, b);

  int                 N = 100'000;
  std::vector<double> fx(N);
  std::generate(std::execution::par_unseq, fx.begin(), fx.end(), [&]() { return f(dis(gen)); });
  return (b - a) * std::accumulate(fx.begin(), fx.end(), 0.0) / N;
}

int main() {
  auto f = [](double x) { return 4 * std::sin(x) + std::cos(x); };

  { // Integrate [ 4 Sin(x) + Cos(x) , {x, 0, Pi}]
    auto_cpu_timer timer;
    std::cout << monte_carlo_integral(f, 0, M_PI) << std::endl;
  }

  { // Integrate [ 4 Sin(x) + Cos(x) , {x, 0, 1+ RandomReal[]}] 并行 2000次
    auto_cpu_timer      timer;
    std::vector<double> bs(2000), results(2000);
    std::generate(std::execution::par_unseq, bs.begin(), bs.end(), []() { return drand48() + 1; });
    std::transform(std::execution::par, bs.begin(), bs.end(), results.begin(), std::bind(monte_carlo_integral, f, 0, _1));
    for (int i = 0; i < bs.size(); ++i) {
      fmt::print("{}: {:.2f}, {:.2f}\n", i+1, bs[i], results[i]);
    }
  }

  return 0;
}
