# 17: AP x metabolite interaction for each of the 249 NMR measures in the main matched cohort.
# Cox models stratified by matched set: exp + BMI + HbA1c + metabolite + exp:metabolite. The metabolite HR in controls is
# exp(b_metabolite); in patients with AP it is exp(b_metabolite + b_interaction).
# Also: biliary and non-biliary matched sets separately and a 3-way test (etiology x AP x metabolite, with exp:biliary);
# sensitivity analyses excluding (overall and in biliary sets) or adjusting for cholesterol-lowering medication, adjusting for time since last meal,
# and using a rank-based inverse normal transformation.
suppressMessages(library(survival))
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
root <- Sys.getenv("PPDM_OUTPUT_DIR", file.path(here, ".."))
d <- read.csv(file.path(root, "cache", "H03_nmr_matched.csv"))
mets <- grep("^p23[4-6][0-9]{2}$", names(d), value = TRUE)

two_group <- function(dd, x, extra = "") {
  dd <- dd[!is.na(dd[[x]]), ]
  if (extra != "") dd <- dd[!is.na(dd[[extra]]), ]
  keep <- ave(dd$exp, dd$mid, FUN = function(z) any(z == 1) & any(z == 0)) == 1
  dd <- dd[keep, ]; dd$x <- dd[[x]]
  f <- as.formula(paste("Surv(T, Y) ~ exp + bmi5 + hba1c5 + x + exp:x", if (extra != "") paste("+", extra) else "", "+ strata(mid)"))
  m <- tryCatch(coxph(f, data = dd), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  b <- coef(m); V <- vcov(m)
  b1 <- b["x"] + b["exp:x"]; se1 <- sqrt(V["x", "x"] + V["exp:x", "exp:x"] + 2 * V["x", "exp:x"]); se0 <- sqrt(V["x", "x"])
  data.frame(n = nrow(dd), n_AP = sum(dd$exp == 1), ev_AP = sum(dd$Y[dd$exp == 1]), ev_ctrl = sum(dd$Y[dd$exp == 0]),
             logHR_ctrl = b["x"], se_ctrl = se0, logHR_AP = b1, se_AP = se1,
             P_ctrl = 2 * pnorm(-abs(b["x"] / se0)), P_AP = 2 * pnorm(-abs(b1 / se1)),
             P_int = summary(m)$coefficients["exp:x", "Pr(>|z|)"], row.names = NULL)
}
run <- function(dd, label, extra = "") {
  out <- do.call(rbind, lapply(mets, function(x) { r <- two_group(dd, x, extra); if (is.null(r)) NULL else cbind(code = x, r) }))
  out$set <- label; out$FDR_int <- p.adjust(out$P_int, "BH"); out$FDR_ctrl <- p.adjust(out$P_ctrl, "BH"); out$FDR_AP <- p.adjust(out$P_AP, "BH")
  cat(sprintf("%s: %d measures; interaction FDR<0.05 %d; control FDR<0.05 %d; AP FDR<0.05 %d\n", label, nrow(out),
              sum(out$FDR_int < 0.05), sum(out$FDR_ctrl < 0.05), sum(out$FDR_AP < 0.05)))
  out
}
res <- rbind(run(d, "all"), run(d[d$biliary == 1, ], "biliary"), run(d[d$biliary == 0, ], "non-biliary"),
             run(d[!is.na(d$chol_med) & d$chol_med == 0, ], "no cholesterol-lowering drugs"), run(d, "adjusted for cholesterol-lowering drugs", extra = "chol_med"),
             run(d[d$biliary == 1 & !is.na(d$chol_med) & d$chol_med == 0, ], "biliary, no cholesterol-lowering drugs"), run(d, "adjusted for fasting time", extra = "fast_h"))
# sensitivity: rank-based inverse normal transformation instead of the log transform
di <- d; for (x in mets) { v <- di[[x]]; ok <- !is.na(v); di[[x]][ok] <- qnorm((rank(v[ok]) - 0.5) / sum(ok)) }
res <- rbind(res, run(di, "inverse normal transformation"))
three <- do.call(rbind, lapply(mets, function(x) {
  dd <- d[!is.na(d[[x]]), ]; dd$x <- dd[[x]]
  m <- tryCatch(coxph(Surv(T, Y) ~ exp + bmi5 + hba1c5 + x + exp:x + exp:biliary + x:biliary + exp:x:biliary + strata(mid), data = dd), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  data.frame(code = x, P_3way = summary(m)$coefficients["exp:x:biliary", "Pr(>|z|)"])
}))
three$FDR_3way <- p.adjust(three$P_3way, "BH")
cat(sprintf("3-way: FDR<0.05 %d; P<0.05 %d\n", sum(three$FDR_3way < 0.05), sum(three$P_3way < 0.05)))
write.csv(res, file.path(root, "results", "H03_nmr_interaction.csv"), row.names = FALSE)
write.csv(three, file.path(root, "results", "H03_nmr_3way.csv"), row.names = FALSE)
