# =========================================================================================
# Drift Diffusion Models: a practical introduction (OPAM 2026 workshop)
# Worked example in R: speed vs. accuracy in lexical decision
# Wagenmakers, Ratcliff, Gomez, & McKoon (2008), Experiment 1
#
# This script runs the full analysis shown in the slides, from the data to the fit checks
# and the interpretation. Run it from top to bottom, or section by section along with the slides.
#
# Requirements: R 3.6 or later and an internet connection for the first install.
# Running time: each model fit takes 1-2 minutes, and the whole script about 10-15 minutes
# on a laptop. The slow steps are marked with "(slow)".
# =========================================================================================


# 0 · Setup -------------------------------------------------------------------------------
# Install driftd (version 0.1.0) from GitHub if it is not installed yet or an older version is
# installed (then restart R before going on). This also installs rtdists, ggplot2 and the other
# packages driftd needs.
if (!requireNamespace("driftd", quietly = TRUE) || packageVersion("driftd") < "0.1.0") {
  # remotes rather than pak: driftd is plain R code, and with build = FALSE it installs on
  # Windows without Rtools
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("JStrueder/driftd@v0.1.0", build = FALSE)
}

library(driftd)
library(ggplot2)

# Colours as on the slides: blue = accuracy instructions, red = speed instructions
cols <- c(accuracy = "#34506B", speed = "#CC0000")


# The data --------------------------------------------------------------------------------
data(speed_acc)                       # ships with driftd

# Word trials with high- or very-low-frequency words. For words, response 1 ("word") is
# correct, so the data are accuracy coded: upper boundary = correct, lower = error.
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
n <- aggregate(cbind(trials = 1, errors = 1 - response) ~
  id + condition + stimulus, data = d, FUN = sum)
with(n, tapply(trials, list(condition, stimulus), min))       # fewest trials in any cell
with(n, tapply(errors, list(condition, stimulus), median))    # median number of errors
range(d$rt)

# RT distributions, all participants pooled: correct RTs above the axis, errors below,
# errors in a lighter shade.(note that high frequency x accuracy cell has very few errors -> something to keep an eye on if fit looks off)
d$outcome <- factor(ifelse(d$response == 1, "correct", "error"))
d$cell <- paste(d$condition, d$outcome)
ggplot(d, aes(rt, fill = cell)) +
  geom_histogram(data = subset(d, outcome == "correct"), aes(y = after_stat(count)),
                 binwidth = 0.05, boundary = 0) +
  geom_histogram(data = subset(d, outcome == "error"), aes(y = -after_stat(count)),
                 binwidth = 0.05, boundary = 0) +
  geom_hline(yintercept = 0) +
  facet_grid(stimulus ~ condition) +   # add id to the facets, or subset(d, id == 1), to see individuals
  scale_fill_manual(values = c("accuracy correct" = "#34506B", "accuracy error" = "#9AA8B5",
                               "speed correct" = "#CC0000", "speed error" = "#EB9999"),
                    guide = "none") +
  scale_y_continuous(labels = abs) + coord_cartesian(xlim = c(0.15, 1.6)) +
  labs(x = "RT (s)", y = "number of trials", title = "All participants pooled") + theme_bw()


# Steps 2-3 · Specify and fit Model 1 (slow) -----------------------------------------------
# v by word frequency, a and Ter (t0) by instruction, z = a/2 (accuracy coding),
# st (st0, variability in Ter) estimated; sv and sz fixed at 0; s = 1.
m1 <- driftd(d, sub_id = "id", response = "response", rt = "rt",
             vary_v = "stimulus", vary_a = "condition", vary_t0 = "condition",
             fix_z = TRUE, fix_st0 = FALSE)


# Step 3, continued · Did the fit converge? -----------------------------------------------
fi <- m1$fit_info
fi[1:5, c("id", "optimizer_runs", "converge", "se_status")]
table(fi$converge)                    # 0 = converged
table(fi$se_status)                   # "ok" = the likelihood has a proper peak


# Step 4 · Check fit: does the model reproduce the data? ----------------------------------
# Crosses: observed (mean of participants); large points: model;
# bars: 95% prediction intervals for individual participants
pred <- predict(m1, seed = 1)
labels <- c(upper = "correct", lower = "error")
# Left on the slide: P(correct) by word frequency and instruction
plot(pred, type = "choice", x_var = "stimulus", color = "condition", response_labels = labels) +
  scale_colour_manual(values = cols, name = "instruction") +
  scale_x_discrete(labels = c(high = "high freq.", very_low = "very low freq."))
# Right on the slide: RT quantiles for very-low-frequency words
plot(pred, select = list(stimulus = "very_low"), color = "condition", response_labels = labels) +
  scale_colour_manual(values = cols, name = "instruction")
# The same for all cells
plot(pred, color = "condition", response_labels = labels) +
  scale_colour_manual(values = cols, name = "instruction")
head(summary(pred), 4)


# Step 4, continued · Can the design recover the parameters? (slow) -----------------------
# Simulates data from each participant's Model 1 estimates (observed trial numbers),
# refits the same model, and compares true and recovered values.
rec <- recover_parameters(m1, seed = 1)
rec
plot(rec)


# Step 4, continued · Would a simpler model fit equally well? (slow) ----------------------
# Refit with one drift rate for both word frequencies, then compare with Model 1:
# AIC and BIC (summed and per participant) and a likelihood-ratio test of the nested models.
m_v0 <- driftd(d, sub_id = "id", response = "response", rt = "rt",
               vary_a = "condition", vary_t0 = "condition",          # v not varied
               fix_z = TRUE, fix_st0 = FALSE)
compare_models(m_v0, m1)

# The same check for boundary separation and non-decision time (two more model fits)
m_a0 <- driftd(d, sub_id = "id", response = "response", rt = "rt",
               vary_v = "stimulus", vary_t0 = "condition",           # a not varied
               fix_z = TRUE, fix_st0 = FALSE)
m_t0 <- driftd(d, sub_id = "id", response = "response", rt = "rt",
               vary_v = "stimulus", vary_a = "condition",            # Ter not varied
               fix_z = TRUE, fix_st0 = FALSE)
compare_models(m_a0, m1)
compare_models(m_t0, m1)


# Step 5 · Interpret: the estimates, one row per participant ------------------------------
est <- m1$individual_estimates
est                      # parameters, convergence and fit (LLE, AIC, BIC) per participant
estimates(m1)            # the same estimates in long format, with standard errors

# Each participant's estimates (light lines) and the means (red)
long <- rbind(
  data.frame(id = est$id, parameter = "v (drift rate)", level = "high freq.", value = est$v_high),
  data.frame(id = est$id, parameter = "v (drift rate)", level = "very low freq.", value = est$v_very_low),
  data.frame(id = est$id, parameter = "a (boundary separation)", level = "accuracy", value = est$a_accuracy),
  data.frame(id = est$id, parameter = "a (boundary separation)", level = "speed", value = est$a_speed),
  data.frame(id = est$id, parameter = "Ter (s)", level = "accuracy", value = est$t0_accuracy),
  data.frame(id = est$id, parameter = "Ter (s)", level = "speed", value = est$t0_speed))
long$parameter <- factor(long$parameter, unique(long$parameter))
ggplot(long, aes(level, value)) +
  geom_line(aes(group = id), colour = "grey60", alpha = 0.6) +
  geom_point(colour = "grey60", alpha = 0.6) +
  stat_summary(aes(group = 1), fun = mean, geom = "line", colour = "#CC0000", linewidth = 1.2) +
  stat_summary(fun = mean, geom = "point", colour = "#CC0000", size = 3) +
  facet_wrap(~ parameter, scales = "free") +
  labs(x = NULL, y = "estimate") + theme_bw()

# How many participants show each effect in the predicted direction?
sum(est$v_high > est$v_very_low)                  # 17 of 17
sum(est$a_accuracy > est$a_speed)                 # 17 of 17
sum(est$t0_accuracy > est$t0_speed)               # 14 of 17
colMeans(est[c("v_high", "v_very_low", "a_accuracy", "a_speed",
               "t0_accuracy", "t0_speed", "st0")])


# Step 5, continued · Are the differences reliable? ---------------------------------------
# Paired t-tests on the individual estimates
t.test(est$a_accuracy, est$a_speed, paired = TRUE)
t.test(est$v_high, est$v_very_low, paired = TRUE)
t.test(est$t0_accuracy, est$t0_speed, paired = TRUE)


# Backup · All Model 1 estimates with standard errors -------------------------------------
estimates(m1)

# For reporting: software versions
packageVersion("driftd")
packageVersion("rtdists")
