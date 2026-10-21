# =========================================================================================
# Drift Diffusion Models: a practical introduction (OPAM 2026 workshop)
# Worked example in R, hierarchical Bayesian version with brms
# Wagenmakers, Ratcliff, Gomez, & McKoon (2008), Experiment 1
#
# This script repeats the analysis of driftd_workshop.R, step by step, but estimates the
# model hierarchically with brms (Buerkner, 2017) instead of fitting each participant
# separately by maximum likelihood. Participants and the group are estimated together, so
# individual estimates are partially pooled toward the group, and condition effects are
# tested on the posterior of the group-level difference instead of with t-tests.
#
# One difference from Model 1 in driftd_workshop.R: the brms wiener family is the
# four-parameter DDM (drift, boundary separation, non-decision time, bias). It has no
# across-trial variability parameters, so st (variability in non-decision time) cannot be
# estimated. Model 1 here is therefore driftd's Model 1 without st. Section 5c compares
# it with the same model fitted per participant in driftd.
#
# Requirements: R 4.0 or later, brms with a working Stan installation (rstan, or cmdstanr),
# and an internet connection for the first install. Installing rstan needs a C++
# toolchain; see https://github.com/stan-dev/rstan/wiki for your system.
# Running time: the model fit (Step 3) is slow, about 1.5 hours with 4 chains on 4 cores; 
# everything else takes a few minutes. The fit is saved to brms_m1.rds in the working directory, so running the
# script again loads it instead of refitting. Delete the file to refit.
# =========================================================================================


# 0 · Setup -------------------------------------------------------------------------------
if (!requireNamespace("brms", quietly = TRUE)) install.packages("brms")   # also installs rstan
# driftd provides the data, and rtdists, used to simulate data for the predictive check
if (!requireNamespace("driftd", quietly = TRUE) || packageVersion("driftd") < "0.1.0") {
  # remotes rather than pak: driftd is plain R code, and with build = FALSE it installs on
  # Windows without Rtools
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("JStrueder/driftd@v0.1.0", build = FALSE)
}

library(brms)
library(ggplot2)
library(driftd)

n_cores <- min(4, parallel::detectCores())   # one core per chain if you have them
cols <- c(accuracy = "#34506B", speed = "#CC0000")


# The data --------------------------------------------------------------------------------
data(speed_acc, package = "driftd")

# Word trials with high- or very-low-frequency words, accuracy coded: response 1 ("word")
# is correct and maps onto the upper boundary.
d <- droplevels(subset(speed_acc, stimulus %in% c("high", "very_low")))
head(d, 4)
nrow(d)                               # 10,462 trials
length(unique(d$id))                  # 17 participants

# Descriptives: means of participant means, with 95% confidence intervals
pm <- merge(
  aggregate(cbind(accuracy = response) ~ id + condition + stimulus, data = d, FUN = mean),
  aggregate(rt ~ id + condition + stimulus, data = subset(d, response == 1), FUN = mean))
desc <- do.call(rbind, lapply(c("accuracy", "rt"), function(measure) {
  s <- aggregate(pm[[measure]], by = pm[c("condition", "stimulus")], FUN = function(x)
    c(mean = mean(x), half = qt(0.975, length(x) - 1) * sd(x) / sqrt(length(x))))
  data.frame(s[c("condition", "stimulus")], measure = measure,
             mean = s$x[, "mean"], lower = s$x[, "mean"] - s$x[, "half"],
             upper = s$x[, "mean"] + s$x[, "half"])
}))
desc$measure <- factor(desc$measure, c("accuracy", "rt"),
                       c("Proportion correct", "Mean correct RT (s)"))
desc

ggplot(desc, aes(stimulus, mean, colour = condition, group = condition)) +
  geom_line(position = position_dodge(0.2)) +
  geom_point(size = 3, position = position_dodge(0.2)) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.1, position = position_dodge(0.2)) +
  facet_wrap(~ measure, scales = "free_y") +
  scale_colour_manual(values = cols, name = "instruction") +
  scale_x_discrete(labels = c(high = "high freq.", very_low = "very low freq.")) +
  labs(x = NULL, y = NULL) + theme_bw() + theme(legend.position = "bottom")


# Step 1 · Prepare: count trials and errors, look at the RT distributions ------------------
# Identical to driftd_workshop.R: the checks do not depend on the estimation method.
n <- aggregate(cbind(trials = 1, errors = 1 - response) ~
  id + condition + stimulus, data = d, FUN = sum)
with(n, tapply(trials, list(condition, stimulus), min))       # fewest trials in any cell
with(n, tapply(errors, list(condition, stimulus), median))    # median number of errors
range(d$rt)

d$outcome <- factor(ifelse(d$response == 1, "correct", "error"))
d$cell <- paste(d$condition, d$outcome)
ggplot(d, aes(rt, fill = cell)) +
  geom_histogram(data = subset(d, outcome == "correct"), aes(y = after_stat(count)),
                 binwidth = 0.05, boundary = 0) +
  geom_histogram(data = subset(d, outcome == "error"), aes(y = -after_stat(count)),
                 binwidth = 0.05, boundary = 0) +
  geom_hline(yintercept = 0) +
  facet_grid(stimulus ~ condition) +
  scale_fill_manual(values = c("accuracy correct" = "#34506B", "accuracy error" = "#9AA8B5",
                               "speed correct" = "#CC0000", "speed error" = "#EB9999"),
                    guide = "none") +
  scale_y_continuous(labels = abs) + coord_cartesian(xlim = c(0.15, 1.6)) +
  labs(x = "RT (s)", y = "number of trials", title = "All participants pooled") + theme_bw()


# Step 2 · Specify Model 1 ----------------------------------------------------------------
# The same mapping as in the workshop slides: v by word frequency, a and Ter by instruction, z = a/2.
# In brms the parameters are called mu (drift), bs (boundary separation), ndt (non-decision
# time) and bias (relative starting point, z/a). dec(response) says which boundary each
# trial ended at: 1 = upper (correct), 0 = lower (error).
#
#  - "0 + stimulus" estimates one drift per frequency (no intercept), so each coefficient
#    is directly the drift in that condition. The same for bs by condition.
#  - "(0 + stimulus | p | id)" gives every participant their own drift per condition.
#    The shared "p" lets the participant effects of all parameters correlate.
#  - bs is estimated on the log scale (link_bs = "log"), which keeps it positive: exp() of
#    a coefficient gives the boundary separation.
#  - bias = 0.5 fixes the starting point at a/2, as fix_z = TRUE does in driftd.
#  - Non-decision time must be shorter than every RT it applies to. Estimated directly,
#    the sampler keeps running into this limit and becomes very slow. So Ter is written as
#    a fraction of the participant's fastest RT in that instruction condition (minrt):
#    ndt = minrt * inv_logit(tfrac), and tfrac is estimated by instruction and participant
#    on the logit scale. This changes how Ter is parameterised, not the model: Ter can
#    still take any value between 0 and the fastest RT.
d$minrt <- ave(d$rt, d$id, d$condition, FUN = min)
f1 <- bf(rt | dec(response) ~ 0 + stimulus + (0 + stimulus | p | id),
         bs ~ 0 + condition + (0 + condition | p | id),
         nlf(ndt ~ minrt * inv_logit(tfrac)),
         tfrac ~ 0 + condition + (0 + condition | p | id),
         bias = 0.5)
fam <- wiener(link_bs = "log", link_ndt = "identity", link_bias = "identity")

# Priors: weakly informative, on the s = 1 scale used throughout the workshop.
#  - drift: normal(2, 2), most mass between about -2 and 6
#  - boundary separation: log a ~ normal(0, 0.5), a between about 0.4 and 2.7
#  - Ter: logit of the fraction ~ normal(2, 1), between about 50% and 98% of the fastest RT
#  - between-participant SDs: half-normal; correlations: LKJ(2), mildly favouring 0
priors <- c(prior(normal(2, 2), class = b),
            prior(normal(0, 0.5), class = b, dpar = bs),
            prior(normal(2, 1), class = b, nlpar = tfrac),
            prior(normal(0, 1), class = sd),
            prior(normal(0, 0.5), class = sd, dpar = bs),
            prior(normal(0, 1), class = sd, nlpar = tfrac),
            prior(lkj(2), class = cor))


# Helpers used below ---------------------------------------------------------------------
# simulate_ddm(): simulate a data set with the design of `data` (same participants, cells
# and trial numbers) from participant parameters `p`, a data frame with one row per
# participant and the columns of `est` below. Used for the posterior predictive check.
simulate_ddm <- function(p, data) {
  cells <- aggregate(cbind(n = rt) ~ id + condition + stimulus, data = data, FUN = length)
  cells <- merge(cells, p, by = "id")
  do.call(rbind, lapply(seq_len(nrow(cells)), function(i) {
    r  <- cells[i, ]
    v  <- if (r$stimulus == "high") r$v_high else r$v_very_low
    a  <- if (r$condition == "accuracy") r$a_accuracy else r$a_speed
    t0 <- if (r$condition == "accuracy") r$t0_accuracy else r$t0_speed
    s  <- rtdists::rdiffusion(r$n, a = a, v = v, t0 = t0, z = a / 2)
    data.frame(r[c("id", "condition", "stimulus")], rt = s$rt,
               response = as.integer(s$response == "upper"))
  }))
}


# Step 3 · Fit Model 1 (slow) -------------------------------------------------------------
# 4 chains of 2,000 iterations, the first 1,000 warm-up. init = 0 starts every chain at
# drift 0, a = 1 and Ter at half the fastest RT, which is valid for every trial.
# file = "brms_m1" saves the fit; running the line again loads it instead of refitting.
m1 <- brm(f1, data = d, family = fam, prior = priors, init = 0,
          chains = 4, cores = n_cores, iter = 2000, warmup = 1000, seed = 1,
          file = "brms_m1")


# Step 4 · Check convergence --------------------------------------------------------------
# For MCMC, convergence means that the chains agree and have explored the posterior:
# R-hat below 1.01 for every parameter, bulk and tail ESS of at least a few hundred, and no
# divergent transitions (Vehtari et al., 2021).
summary(m1)                                   # Rhat, Bulk_ESS and Tail_ESS columns
max(rhat(m1), na.rm = TRUE)                   # should be below 1.01
conv <- posterior::summarise_draws(as_draws_df(m1, variable = c("^b_", "^sd_", "^cor_"),
                                               regex = TRUE))
conv[order(conv$ess_bulk), c("variable", "rhat", "ess_bulk", "ess_tail")][1:5, ]  # lowest ESS
np <- nuts_params(m1)
sum(np$Value[np$Parameter == "divergent__"])  # number of divergent transitions: should be 0
# Trace plots: the four chains should overlap and look like stationary noise
mcmc_plot(m1, type = "trace", variable = "^b_", regex = TRUE)


# Step 5 · Interpret: participant and group estimates ------------------------------------
# 5a. Participant level: each participant's parameters, one row per participant, in the
#     same layout as m1$individual_estimates in driftd (posterior means). coef() returns
#     the draws of each participant's coefficients, which are then put on the scale of the
#     parameters: exp() for a, the fraction of the fastest RT for Ter.
cf <- coef(m1, summary = FALSE)$id            # draws x participant x coefficient
ids <- dimnames(cf)[[2]]
minrt <- function(cond) d$minrt[match(paste(ids, cond), paste(d$id, d$condition))]
pars <- list(
  v_high      = cf[, , "stimulushigh"],
  v_very_low  = cf[, , "stimulusvery_low"],
  a_accuracy  = exp(cf[, , "bs_conditionaccuracy"]),
  a_speed     = exp(cf[, , "bs_conditionspeed"]),
  t0_accuracy = sweep(plogis(cf[, , "tfrac_conditionaccuracy"]), 2, minrt("accuracy"), "*"),
  t0_speed    = sweep(plogis(cf[, , "tfrac_conditionspeed"]), 2, minrt("speed"), "*"))
est <- data.frame(id = ids, lapply(pars, colMeans))
est

# Without st, Ter sits just below each participant's fastest RT in each condition (here at
# 84% to 97% of it), so a single very fast trial can pull Ter down.

# 5b. Group level. For v and a, the population-level coefficients describe the typical
#     participant. Ter is defined relative to each participant's fastest RT, so its group
#     value is the average over the 17 participants instead.
draws <- as_draws_df(m1)
group <- data.frame(
  v_high      = draws$b_stimulushigh,
  v_very_low  = draws$b_stimulusvery_low,
  a_accuracy  = exp(draws$b_bs_conditionaccuracy),
  a_speed     = exp(draws$b_bs_conditionspeed),
  t0_accuracy = rowMeans(pars$t0_accuracy),
  t0_speed    = rowMeans(pars$t0_speed))
posterior::summarise_draws(posterior::as_draws_df(group), "mean", "sd",
                           ~ quantile(.x, c(0.025, 0.975)))

# Each participant's estimates (light lines) and the group values (red)
long <- rbind(
  data.frame(id = est$id, parameter = "v (drift rate)", level = "high freq.", value = est$v_high),
  data.frame(id = est$id, parameter = "v (drift rate)", level = "very low freq.", value = est$v_very_low),
  data.frame(id = est$id, parameter = "a (boundary separation)", level = "accuracy", value = est$a_accuracy),
  data.frame(id = est$id, parameter = "a (boundary separation)", level = "speed", value = est$a_speed),
  data.frame(id = est$id, parameter = "Ter (s)", level = "accuracy", value = est$t0_accuracy),
  data.frame(id = est$id, parameter = "Ter (s)", level = "speed", value = est$t0_speed))
long$parameter <- factor(long$parameter, unique(long$parameter))
grp <- data.frame(parameter = factor(rep(levels(long$parameter), each = 2), levels(long$parameter)),
                  level = c("high freq.", "very low freq.", "accuracy", "speed", "accuracy", "speed"),
                  value = colMeans(group))
ggplot(long, aes(level, value)) +
  geom_line(aes(group = id), colour = "grey60", alpha = 0.6) +
  geom_point(colour = "grey60", alpha = 0.6) +
  geom_line(data = grp, aes(group = 1), colour = "#CC0000", linewidth = 1.2) +
  geom_point(data = grp, colour = "#CC0000", size = 3) +
  facet_wrap(~ parameter, scales = "free") +
  labs(x = NULL, y = "estimate (posterior mean)") + theme_bw()

# How many participants show each effect in the predicted direction?
sum(est$v_high > est$v_very_low)
sum(est$a_accuracy > est$a_speed)
sum(est$t0_accuracy > est$t0_speed)

# 5c. Partial pooling: compare with the same model fitted per participant by maximum
#     likelihood in driftd (no st, as in brms). Toypically, hierarchical estimates are pulled toward the group.
#     With around 600 trials per participant here, each participant's own data dominate, so the points sit
#     close to the dashed line and pooling changes little. Would make a larger difference with fewer trials.
library(driftd)
ml <- driftd(d, sub_id = "id", response = "response", rt = "rt",
             vary_v = "stimulus", vary_a = "condition", vary_t0 = "condition",
             fix_z = TRUE, ask = FALSE)$individual_estimates
shrink <- rbind(
  data.frame(parameter = "v high freq.", ml = ml$v_high, brms = est$v_high),
  data.frame(parameter = "v very low freq.", ml = ml$v_very_low, brms = est$v_very_low),
  data.frame(parameter = "a accuracy", ml = ml$a_accuracy, brms = est$a_accuracy),
  data.frame(parameter = "a speed", ml = ml$a_speed, brms = est$a_speed),
  data.frame(parameter = "Ter accuracy", ml = ml$t0_accuracy, brms = est$t0_accuracy),
  data.frame(parameter = "Ter speed", ml = ml$t0_speed, brms = est$t0_speed))
shrink$parameter <- factor(shrink$parameter, unique(shrink$parameter))
ggplot(shrink, aes(ml, brms)) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  geom_point(colour = "#34506B") +
  facet_wrap(~ parameter, scales = "free") +
  labs(x = "per-participant maximum likelihood (driftd)",
       y = "hierarchical posterior mean (brms)") + theme_bw()


# Step 5, continued · Is a condition effect reliable? -------------------------------------
# 1. Does the average participant show it? In a hierarchical model the group-level
#    difference is itself estimated, so there is no t-test: usually report the posterior of the
#    difference, its 95% credible interval and the posterior probability of the predicted
#    direction (Post.Prob). Evid.Ratio is the posterior odds of that direction. alpha =
#    0.025 makes brms report a 95% interval for these one-sided hypotheses.
hypothesis(m1, "stimulushigh - stimulusvery_low > 0", alpha = 0.025)                # drift
hypothesis(m1, "exp(bs_conditionaccuracy) - exp(bs_conditionspeed) > 0", alpha = 0.025)
# Ter: the average over participants of accuracy minus speed (seconds), computed from the
# draws because Ter is defined relative to each participant's fastest RT
ter_diff <- rowMeans(pars$t0_accuracy - pars$t0_speed)
c(estimate = mean(ter_diff), quantile(ter_diff, c(0.025, 0.975)),
  post_prob = mean(ter_diff > 0))

# 2. Does each participant show it? For every participant, the posterior probability that
#    their own effect goes in the predicted direction. A probability above .95 is the
#    Bayesian counterpart of "significant for this participant" (n_significant in driftd).
p_dir <- data.frame(
  id = est$id,
  v = colMeans(pars$v_high > pars$v_very_low),
  a = colMeans(pars$a_accuracy > pars$a_speed),
  Ter = colMeans(pars$t0_accuracy > pars$t0_speed))
p_dir
colSums(p_dir[-1] > 0.95)


# Step 6 · Before reporting: does the model reproduce the data? ---------------------------
# Posterior predictive check: simulate data sets from the posterior (here 100 draws, each
# with the observed participants, cells and trial numbers), compute the same statistics as
# for the data, and see whether the observed values fall within the predicted range.
# (posterior_predict(m1, negative_rt = TRUE) does the same, but is slow for this family.)
set.seed(1)
draw_ids <- sample(ndraws(m1), 100)
yrep <- lapply(draw_ids, function(k)
  simulate_ddm(data.frame(id = est$id, lapply(pars, function(x) x[k, ])), d))

# Statistics per participant and cell: P(correct) and the .1, .3, .5, .7, .9 quantiles of
# correct and error RTs (error quantiles only for cells with at least 5 errors), then the
# mean over participants, as in driftd's plots.
probs <- c(0.1, 0.3, 0.5, 0.7, 0.9)
cell_stats <- function(data) {
  key <- interaction(data$id, data$condition, data$stimulus, drop = TRUE)
  per <- lapply(split(seq_len(nrow(data)), key), function(i) {
    rt <- data$rt[i]; correct <- data$response[i] == 1
    q <- function(x) if (length(x) >= 5) quantile(x, probs, names = FALSE) else rep(NA, 5)
    c(p_correct = mean(correct), q_correct = q(rt[correct]), q_error = q(rt[!correct]))
  })
  per <- data.frame(do.call(rbind, per),
                    data[match(names(per), key), c("condition", "stimulus")])
  aggregate(. ~ condition + stimulus, data = per, FUN = mean, na.rm = TRUE,
            na.action = na.pass)
}
obs <- cell_stats(d)
rep <- do.call(rbind, Map(cbind, lapply(yrep, cell_stats), draw = seq_along(yrep)))

# Long format: one row per cell, statistic and draw
stat_cols <- setdiff(names(obs), c("condition", "stimulus"))
to_long <- function(x) {
  out <- do.call(rbind, lapply(stat_cols, function(s)
    data.frame(x[c("condition", "stimulus")], stat = s, value = x[[s]],
               draw = if (is.null(x$draw)) NA else x$draw)))
  out$response <- ifelse(grepl("error", out$stat), "error",
                         ifelse(out$stat == "p_correct", NA, "correct"))
  out$quantile <- as.numeric(sub(".*_[a-z]+", "", out$stat)) # 1 to 5
  out
}
obs_l <- to_long(obs)
pred_l <- to_long(rep)
pred_s <- aggregate(value ~ condition + stimulus + stat, data = pred_l,
                    FUN = function(x) c(mid = median(x), lower = quantile(x, 0.025, names = FALSE),
                                        upper = quantile(x, 0.975, names = FALSE)))
pred_s <- data.frame(pred_s[c("condition", "stimulus", "stat")], pred_s$value)
ppc <- merge(obs_l, pred_s, by = c("condition", "stimulus", "stat"))

# Left on the slide: P(correct). Crosses: observed (mean of participants); points and bars:
# median and 95% interval of the same statistic in the posterior predictive data sets.
ggplot(subset(ppc, stat == "p_correct"), aes(stimulus, colour = condition, group = condition)) +
  geom_pointrange(aes(y = mid, ymin = lower, ymax = upper), position = position_dodge(0.3)) +
  geom_point(aes(y = value), shape = 4, size = 4, stroke = 1.2, colour = "black",
             position = position_dodge(0.3)) +
  scale_colour_manual(values = cols, name = "instruction") +
  scale_x_discrete(labels = c(high = "high freq.", very_low = "very low freq.")) +
  labs(x = NULL, y = "P(correct)") + theme_bw()

# Right on the slide: RT quantiles of correct (top) and error (bottom) responses, all cells
ggplot(subset(ppc, stat != "p_correct"), aes(quantile, colour = condition, group = condition)) +
  geom_pointrange(aes(y = mid, ymin = lower, ymax = upper), position = position_dodge(0.3)) +
  geom_point(aes(y = value), shape = 4, size = 3, stroke = 1, colour = "black",
             position = position_dodge(0.3)) +
  facet_grid(response ~ stimulus, labeller = labeller(
    stimulus = c(high = "high freq.", very_low = "very low freq."))) +
  scale_colour_manual(values = cols, name = "instruction") +
  scale_x_continuous(breaks = 1:5, labels = probs) +
  labs(x = "RT quantile", y = "RT (s)") + theme_bw()


# For reporting: software versions --------------------------------------------------------
packageVersion("brms")
packageVersion("rstan")
citation("brms")
