#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <random>
#include <cmath>

#include "../utils/ch_18_util.h"

#include <matplot/matplot.h>
using namespace matplot;

double F(double x, double p) {
     if( x < 0.0 )
    	 return 0.0;
     else {
    	 if(x > 1.0)
    		 return 1.0;
    	 else
    		 return (1.0 - p);
     }
}


double DF(double x, double n) {
    //return 0 if x < 1 else 1 if x > n else torch.floor(x) / n
    if( x < 1.0 )
    	return 0.0;
    else {
    	if( x > n )
    		return 1.0;
    	else
    		return std::floor(x) / n;
    }
}

double CF(double x, double a, double b) {
    //return 0 if x < a else 1 if x > b else (x - a) / (b - a)
	if( x < a) {
		return 0.0;
	} else {
		if( x > b ) {
			return 1.0;
		} else {
			return (x - a)/(b - a);
		}
	}
}

// Compute binomial coefficient
int64_t binom(int n, int k) {
	int64_t comb = 1;
    int x = std::min(k, n - k);
    for(int i = 0; i < x; i++ )
        comb = comb * static_cast<int64_t>((n - i) / (i + 1));
    return comb;
}

// to avoid overflow
// std::pow(p, i) * std::pow((1-p), (ns[j]-i)) * binom(ns[j], i);
double binom2(int n, int k, double p) {

	long double comb = 1.0*std::pow(p, k) * std::pow((1-p), (n-k));

    int x = std::min(k, n - k);
    for(int j = 0; j < x; j++ ) {
    	comb = comb * (static_cast<double>((n - j) / (j + 1)));
    }

    return (double) comb;
}


double BF(double x, double n, std::vector<double> cmf) {
    if(x < 0.0)
    	return 0.0;
    else {
    	if( x > n )
    		return 1.0;
    	else
    		return cmf[int(x)];
    }
}


double phi(double x, double mu, double sigma) {
    return (1.0 + std::erf((x - mu) / (sigma * std::sqrt(2.)))) / 2.0;
}



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(1000);

	// --------------------------------
	// Bernoulli
	// --------------------------------

	double p = 0.3;
	std::vector<double> x = {0.0, 1.0};
	std::vector<double> y = {1.0 - p, p};

	auto h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	auto ax1 = h->nexttile();
	matplot::stem(ax1, x, y, "filled")->line_width(3);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "p.m.f.");
    matplot::title("Bernoulli");
    matplot::show();

	// Now, let us plot the cumulative distribution function :eqref:eq_bernoulli_cdf
	x.clear();// = torch::arange(-1, 2, 0.01)
	for(double i = -1.0; i <= 2.0; i += 0.01)
		x.push_back(i);

	y.clear();
	for(auto& j : x) {
		y.push_back(F(j, p));
	}

	h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	ax1 = h->nexttile();
	matplot::plot(ax1, x, y, "b")->line_width(2);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "c.d.f.");
    matplot::title("Bernoulli");
    matplot::show();

	// We can sample an array of arbitrary shape from a Bernoulli random variable as follows
	std::cout << 1*(torch::rand({10, 10}) < p) << "\n";

	// ---------------------------------
	// Discrete Uniform
	// ---------------------------------
	double n = 5.0;
	x.clear(); y.clear();
	for(double i = 1.0; i <= n; i += 1.0) {
		x.push_back(i);
		y.push_back(1/n);
	}

	h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	ax1 = h->nexttile();
	matplot::stem(ax1, x, y, "filled")->line_width(3);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "p.m.f.");
    matplot::title("Discrete Uniform");
    matplot::show();

	// Now, let us plot the cumulative distribution function
	x.clear();
	for(double i = -1.0; i <= 6.0; i += 0.01)
		x.push_back(i);

	y.clear();
	for(auto& j : x) {
		y.push_back(DF(j, n));
	}

	h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	ax1 = h->nexttile();
	matplot::plot(ax1, x, y, "b")->line_width(2);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "c.d.f.");
    matplot::title("Discrete Uniform");
    matplot::show();

	// We can sample an array of arbitrary shape from a discrete uniform random variable as follows.
	std::cout << torch::randint(1, n, {10, 10}) << "\n";

	// ----------------------------------
	// Continuous Uniform
	// ----------------------------------
	double a = 1.0, b = 3.0;
	x.clear(); y.clear();
	for(double i = 0.0; i <= 4.0; i += 0.01) {
		x.push_back(i);
		if( i > a && i < b)
			y.push_back(1.0);
		else
			y.push_back(0.0);
	}

	h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	ax1 = h->nexttile();
	matplot::plot(ax1, x, y, "b")->line_width(2);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "p.m.f.");
    matplot::title("Continuous Uniform");
    matplot::show();

	// Now, let us plot the cumulative distribution function
	y.clear();
	for(auto& j : x) {
		y.push_back(CF(j, a, b));
	}

	h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	ax1 = h->nexttile();
	matplot::plot(ax1, x, y, "b")->line_width(2);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "c.d.f.");
    matplot::title("Continuous Uniform");
    matplot::show();

	// We can sample an array of arbitrary shape from a uniform random variable as follows.
	std::cout << (b - a) * torch::rand({10, 10}) + a << "\n";

	// -----------------------------
	// Binomial
	// -----------------------------
	int N = 10;
	p = 0.2;
	x.clear();
	y.clear();

	for( int k = 0; k < (N + 1); k++ ) {
		x.push_back(k*1.0);
		double t = std::pow(p, k) * std::pow((1-p), (N-k)) * binom(N, k);
		y.push_back(t);
	}

	//pmf = torch.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

	h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	ax1 = h->nexttile();
	matplot::stem(ax1, x, y, "filled")->line_width(3);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "p.m.f.");
    matplot::title("Binomial");
    matplot::show();

	// Now, let us plot the cumulative distribution function
	auto pmf = torch::from_blob(y.data(), {(int64_t)y.size()}, at::TensorOptions(torch::kDouble)).clone();
	auto xx = torch::arange(-1, 11, 0.01).to(torch::kDouble);
	auto cmf = torch::cumsum(pmf, 0).to(torch::kDouble);
	std::cout << cmf[0] << "\n";

	// convert tensor to vector
	std::vector<double> xa(xx.numel());
	std::memcpy(&(xa[0]), xx.data_ptr<double>(),sizeof(double)*xx.numel());
	std::cout << xa[0] << "\n";

	std::vector<double> ya(cmf.numel());
	std::memcpy(&(ya[0]), cmf.data_ptr<double>(),sizeof(double)*cmf.numel());

	y.clear();
	for( auto& i : xa ) {
		y.push_back(BF(i, N*1.0, ya));
	}

	h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	ax1 = h->nexttile();
	matplot::plot(ax1, xa, y, "b")->line_width(2);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "c.d.f.");
    matplot::title("Binomial");
    matplot::show();

	// This follows from the linearity of expected value over the sum of n Bernoulli random variables
	//auto m = torch::distributions::binomial::Binomial(n, p);
	//m.sample({10, 10});
	//auto m = torch::binomial(torch::tensor({10}), torch::tensor({0.2}));
	//std::cout << m.values() << "\n";

	std::random_device rd;
	std::mt19937 gen(rd());
	// perform 4 trials, each succeeds 1 in 2 times
	std::binomial_distribution<> d(N, p);
	std::vector<double> B;
	for( int i = 0; i < 100; i++ )
		B.push_back(d(gen) * 1.0);

	std::cout << B.size() << "\n";

	auto m = torch::from_blob(B.data(), {10, 10}, at::TensorOptions(torch::kDouble)).clone();
	std::cout << m << "\n";

	// ---------------------------
	// Poisson
	// ---------------------------
	double lam = 5.0;
	std::vector<double> xs;
	y.clear();

	std::cout <<"tgamma(3): " << std::tgamma(2 + 1) << '\n';

	for( int i = 0; i < 20; i++ ) {
		xs.push_back(i*1.0);
		y.push_back(std::exp(-1.0*lam)*std::pow(lam, i)/std::tgamma(i + 1));
	}

	h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	ax1 = h->nexttile();
	matplot::stem(ax1, xs, y, "filled")->line_width(3);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "p.m.f.");
    matplot::title("Poisson");
    matplot::show();

	//x = torch.arange(-1, 21, 0.01)
	auto pm = torch::from_blob(y.data(), {(int64_t)y.size()}, at::TensorOptions(torch::kDouble)).clone();
	cmf = torch::cumsum(pm, 0).to(torch::kDouble);

	std::vector<double> ym(cmf.numel());
	std::memcpy(&(ym[0]), cmf.data_ptr<double>(),sizeof(double)*cmf.numel());
	x.clear();
	y.clear();
	for(double i = -1.0; i <= 21.0; i += 0.01) {
		x.push_back(i);
		y.push_back(BF(i, N, ym));
	}

	h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	ax1 = h->nexttile();
	matplot::plot(ax1, x, y, "b")->line_width(2);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "c.d.f.");
    matplot::title("Poisson");
    matplot::show();

	// the means and variances are particularly concise. This can be sampled as follows.
	std::poisson_distribution<> pd(lam);
	std::vector<double> P;
	for( int i = 0; i < 100; i++ )
		P.push_back(pd(gen) * 1.0);

	m = torch::from_blob(P.data(), {10, 10}, at::TensorOptions(torch::kDouble)).clone();
	std::cout << m << "\n";

	// --------------------------------
	// Gaussian
	// --------------------------------

    auto f = figure(true);
//    std::cout << "W = " << f->width() << " H = " << f->height() << '\n';
    f->size(400, 340);
    f->width(f->width() * 2);
    f->height(f->height() * 2);
    f->x_position(0);
    f->y_position(0);

    matplot::hold(true);

	p = 0.2;
	std::vector<int> ns = {1, 10, 90, 300};

	for(int j = 0; j < 4; j++ ) {
	    x.clear();
	    y.clear();

	    for( int i = 0; i < (ns[j] + 1); i++ ) {
	    	//[(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p))) for i in range(n + 1)]
	    	double tx = (i - ns[j]*p)/std::sqrt(ns[j]*p*(1-p));

	    	x.push_back(tx);
	    	//pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
	    	y.push_back(binom2(ns[j], i, p));
	    }

	    matplot::subplot(1, 4, j);
	    matplot::xlim({-4.0, 4.0});
	    matplot::stem(x, y, "filled")->line_width(3);
	    matplot::title("n = " + std::to_string(ns[j]));
	    matplot::xlabel("x");

	    if( j == 0 ) matplot::ylabel("p.m.f.");
	    f->draw();
	}
	matplot::show();

	// Let us first plot the probability density function
	double mu = 0.0, sigma = 1.0;

	//x = torch.arange(-3, 3, 0.01)
	//p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(-(x - mu)**2 / (2 * sigma**2))
	x.clear();
	y.clear();
	for(double i = -3.0; i <= 3.0; i += 0.01) {
		x.push_back(i);
		double p = 1/std::sqrt(2*M_PI*std::pow(sigma, 2))
				* std::exp(-1.0*std::pow((i-mu),2.0)/(2*std::pow(sigma, 2)));
		y.push_back(p);
	}

	h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	ax1 = h->nexttile();
	matplot::plot(ax1, x, y, "b")->line_width(2);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "c.d.f.");
    matplot::title("Gaussian");
    matplot::show();

	// ----------------------------------
	// Phi - function for a standard normal distribution
	// ----------------------------------
	y.clear();

	for(auto& i : x )
		y.push_back(phi(i, mu, sigma));

	h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	ax1 = h->nexttile();
	matplot::plot(ax1, x, y, "b")->line_width(2);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "c.d.f.");
    matplot::title("Phi");
    matplot::show();

	// We can sample from the Gaussian (or standard normal) distribution as shown below.
	std::cout << torch::normal(mu, sigma, {10, 10}) << "\n";

	std::cout << "Done!\n";
}
