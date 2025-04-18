from scipy.stats import norm

# Estimates the probability that observed data from a hidden Markov model reflects a change 
# point in the underlying latent state, via a likelihood ratio (technically, posterior odds)
# incorporating a prior best guess, a given Gaussian emission process dispersion (which may
# reflect both prior uncertainty and the generative noise), and a given hazard rate. 
#
# Inputs:
#    obs    - The observation data in interval specified by bnds
#    pred   - The prior best guess at a latent state
#    obs_sd - Observation uncertainty (combination of emission and prior uncertainty)
#    hazard - The prior (unconditional, marginal) probability of observing a change-point
#    bnds   - The interval in which it is possible for observations to fall as [lower, upper]
#
# Returns:
#    cpp    - An estimated change-point probability

def get_cpp(obs, pred, obs_sd, hazard, bnds):
    #  Probability of the data given our state estimate and observation SD estimate
    obs_prob = norm.pdf(obs, pred, obs_sd)

    # Normalize to correct for probability outside of range.
    cdf_min  = norm.cdf(bnds[0], pred, obs_sd)
    cdf_max  = norm.cdf(bnds[1], pred, obs_sd)
    obs_prob = obs_prob/(cdf_max - cdf_min)

    # Uniform prior on outcome, conditional on CPP
    like_cp = 1/(bnds[1]-bnds[0])

    # Likelihood ratio in favor of being CP
    like_ratio = like_cp/obs_prob

    # Hazard ratio P(CP)/P(~CP)
    hr = hazard/(1-hazard)

    # One-step optimal CPP estimate as posterior odds
    cr_bayes = like_ratio * hr

    # Convert from odds P(CP) vs P(~CP) odds to probability
    cpp = cr_bayes / (cr_bayes + 1)

    return cpp