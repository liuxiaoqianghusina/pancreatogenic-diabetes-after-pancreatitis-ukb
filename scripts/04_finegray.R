# 04: Fine-Gray competing-risk models (death as competing event), robust SEs clustered on matched set.
# Reads cache/matched_main_for_R.csv written by 03_sensitivity.py; PPDM_OUTPUT_DIR sets the output root.
suppressMessages(library(survival))
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
root <- Sys.getenv("PPDM_OUTPUT_DIR", file.path(here, ".."))
d <- read.csv(file.path(root, "cache", "matched_main_for_R.csv"))
d$status <- factor(d$E, levels = c(0, 1, 2), labels = c("censor", "dm", "death"))
d$id <- seq_len(nrow(d))
fit_fg <- function(dd) {
  fg <- finegray(Surv(T, status) ~ ., data = dd[, c("T", "status", "exp", "hba1c", "male", "bmi", "mid", "id")], etype = "dm")
  m <- coxph(Surv(fgstart, fgstop, fgstatus) ~ exp + hba1c + male + bmi + cluster(mid), weights = fgwt, data = fg)
  s <- summary(m)$conf.int["exp", ]
  sprintf("%.2f (%.2f–%.2f)", s[1], s[3], s[4])
}
out <- data.frame(group = c("All", "HbA1c <36", "HbA1c 36-38", "HbA1c 39-41", "HbA1c 42-47"), SHR = NA)
out$SHR[1] <- fit_fg(d)
for (i in 2:5) out$SHR[i] <- fit_fg(d[d$hcat == sub("HbA1c ", "", out$group[i]), ])
print(out)
write.csv(out, file.path(root, "results", "F04_finegray.csv"), row.names = FALSE, fileEncoding = "UTF-8")
