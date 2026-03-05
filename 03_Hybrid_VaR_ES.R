library(data.table)
library(ggplot2)

# se for AAPL -> AAPL_GARCH_LSTM.csv

df_full <- fread("AMZN_GARCH_LSTM_10.csv")
df_full[, Date := as.IDate(Date)]
setorder(df_full, Date)

returns <- as.numeric(df_full$Returns)
sigma2_hat <- as.numeric(df_full$RV_hat)

n_ins <- sum(df_full$Set == "INS")
n_oos <- sum(df_full$Set == "OOS")  

stopifnot(n_ins > 50, n_oos > 0)

VaR_1 <- VaR_5 <- ES_1 <- ES_5 <- matrix(NA_real_, nrow = n_oos, ncol = 1,
                                         dimnames = list(NULL, "LSTM"))
q01 <- q05 <- rep(NA_real_, n_oos)
r_oos <- rep(NA_real_, n_oos)
mu_oos <- rep(NA_real_, n_oos)

for (i in 1:n_oos) {
  print(i)
  
  returns_window <- returns[i:(i + n_ins - 1)]
  mu <- mean(returns_window)
  mu_oos[i] <- mu
  
  returns_c <- scale(returns_window, scale = FALSE)
  
  acf1 <- tryCatch(
    acf(as.numeric(returns_c), plot = FALSE)$acf[2],
    error = function(e) NA_real_
  )
  
  if (!is.na(acf1) && abs(acf1) > 2/sqrt(length(returns_c))) {
    
    ar_fit <- tryCatch(
      ar.yw(as.numeric(returns_c), order.max = 3, aic = TRUE, se.fit = FALSE),
      error = function(e) NULL
    )
    
    if (!is.null(ar_fit)) {
      r <- as.numeric(ar_fit$resid)
      
      if (anyNA(r)) {
        idx <- is.na(r)
        r[idx] <- as.numeric(returns_c)[idx]
      }
      
      if (all(is.finite(r)) && sd(r) > 1e-10) {
        returns_c <- r
      } else {
        returns_c <- as.numeric(returns_c)
      }
    } else {
      returns_c <- as.numeric(returns_c)
    }
    
  } else {
    returns_c <- as.numeric(returns_c)
  }
  
  sigma2_window <- sigma2_hat[i:(i + n_ins - 1)]
  sigma2_next   <- sigma2_hat[i + n_ins]
  
  res_LSTM <- returns_c / sqrt(sigma2_window)
  
  q01[i] <- as.numeric(quantile(res_LSTM, 0.01, na.rm = TRUE))
  q05[i] <- as.numeric(quantile(res_LSTM, 0.05, na.rm = TRUE))
  
  VaR_1[i, "LSTM"] <- mu + sqrt(sigma2_next) * q01[i]
  VaR_5[i, "LSTM"] <- mu + sqrt(sigma2_next) * q05[i]
  
  ES_1[i, "LSTM"] <- mean(returns_window[returns_window < VaR_1[i, "LSTM"]], na.rm = TRUE)
  ES_5[i, "LSTM"] <- mean(returns_window[returns_window < VaR_5[i, "LSTM"]], na.rm = TRUE)
  
  r_oos[i] <- returns[i + n_ins]
}

idx_next <- (1:n_oos) + n_ins

out <- data.table(
  Date   = df_full$Date[idx_next],
  Return = r_oos,
  mu     = mu_oos,
  q01    = q01,
  q05    = q05,
  RV_hat = sigma2_hat[idx_next],
  VaR_1  = VaR_1[, "LSTM"],
  ES_1   = ES_1[, "LSTM"],
  VaR_5  = VaR_5[, "LSTM"],
  ES_5   = ES_5[, "LSTM"]
)

out[, hit_1 := as.integer(Return < VaR_1)]
out[, hit_5 := as.integer(Return < VaR_5)]

fwrite(out, "AMZN_GARCH_LSTM_VaR_ES_10.csv")

cat("\nHit rate 1%:", mean(out$hit_1, na.rm = TRUE), "\n")
cat("Hit rate 5%:", mean(out$hit_5, na.rm = TRUE), "\n")
print(out[1:5])

out[, Date := as.Date(Date)]

p1 <- ggplot(out, aes(x = Date)) +
  geom_line(aes(y = Return), linewidth = 0.4) +
  geom_line(aes(y = VaR_1), color = 'red', linewidth = 0.6) +
  labs(
    title = "Retorno OoS vs VaR 1% (LSTM)",
    x = NULL, y = "Return / VaR"
  ) +
  theme_minimal()

p5 <- ggplot(out, aes(x = Date)) +
  geom_line(aes(y = Return), linewidth = 0.4) +
  geom_line(aes(y = VaR_5), color = 'red', linewidth = 0.6) +
  labs(
    title = "Retorno OoS vs VaR 5% (LSTM)",
    x = NULL, y = "Return / VaR"
  ) +
  theme_minimal()

p1_hits <- p1 +
  geom_point(data = out[hit_1 == 1], aes(y = Return), size = 1.2)

p5_hits <- p5 +
  geom_point(data = out[hit_5 == 1], aes(y = Return), size = 1.2)

print(p1_hits)
print(p5_hits)
