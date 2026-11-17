# BONUS: Suggested Implementation Plan

## 1. Observed limitations of the current implementation

In the experiment (20,000 trials, bandit rewards [1, 2, 3, 4]), Thompson Sampling clearly outperformed Epsilon-Greedy: its cumulative regret was about 26, compared with about 6,181 for Epsilon-Greedy. The main reasons are:

- With epsilon decaying as 1/t, Epsilon-Greedy almost stops exploring after the first few trials. It pulled the two worst arms only 2 and 3 times and then spent thousands of pulls on the second-best arm, because its estimates for the best arm were never tested enough.
- Each algorithm is run only once, so the result depends on the random seed. A single lucky or unlucky early pull can change the outcome of Epsilon-Greedy considerably.
- The reward precision is assumed to be known, which is rarely the case in a real A/B test.

## 2. Proposed improvements

a) **Slower epsilon decay.** Use epsilon_t = min(1, c*K / t) or epsilon_t = eps0 / sqrt(t), where K is the number of arms. This keeps exploring longer and makes it much more likely that the best arm is found.

b) **Optimistic initialization.** Start every arm's estimated mean at a high value so that each arm is tried several times before the algorithm commits to one.

c) **Multiple runs.** Repeat each experiment with 50-100 different random seeds and plot the average cumulative regret with a confidence band. This shows whether the difference between the algorithms is real and not caused by one seed.

d) **Unknown variance.** Replace the known-precision Gaussian model in Thompson Sampling with a Normal-Gamma posterior, so that both the mean and the precision of each arm are learned from data.

e) **Additional baseline.** Add UCB1 as a third algorithm to make the comparison more complete.

f) **Better visualization.** Plot cumulative regret on a log scale. A good algorithm should show regret growing roughly logarithmically in the number of trials.

g) **Code quality.** Move the shared logic (pull loop, bookkeeping, reporting) into the Bandit base class to avoid duplication, store all settings (rewards, number of trials, epsilon, precision, seed) in one configuration object, and add unit tests, for example checking that the posterior mean in Thompson Sampling converges to the true arm mean.

## 3. Expected result

These changes should make Epsilon-Greedy much more reliable, make the comparison statistically meaningful, and make the code easier to maintain and extend.
