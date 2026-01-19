#############################################################################
#####                       Empirical Application                       #####
#############################################################################
library(dplyr)
library(readxl)
library(stringr)
library(rugarch)
library(GAS)
library(MSGARCH)

# Data
data <- read_excel("APPLE_DATA.xlsx", na = c("", "-", NA)) %>%
  mutate(
    RETURNS_APPLE = as.numeric(RETURNS_APPLE),
    RV_APPLE      = na_if(as.numeric(RV_APPLE), 0)
  ) %>%
  filter(!is.na(RV_APPLE), !is.na(RETURNS_APPLE))

returns <- data$RETURNS_APPLE

ins <- 2500
oos <- nrow(data) - ins

# Specs
garch_spec <- ugarchspec(variance.model = list(model = 'sGARCH', garchOrder = c(1, 1)), mean.model = list(armaOrder = c(0, 0), include.mean = FALSE), distribution.model = "std")
gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity", GASPar = list(locate = FALSE, scale = TRUE, shape = FALSE))
msgarch_spec <- CreateSpec(variance.spec = list(model = c("sGARCH", "sGARCH")), switch.spec = list(do.mix = FALSE), distribution.spec = list(distribution = c("std", "std")), constraint.spec = list(regime.const = c("nu")))

# Matrix
vol_ins <- matrix(NA_real_, nrow = nrow(data), ncol = 3, dimnames = list(NULL, c("GARCH","GAS","MSGARCH")))

# InS
returns_c <- scale(returns[1:ins], scale = FALSE)

fit_GARCH <- ugarchfit(garch_spec, returns_c, solver = "hybrid")
fit_GAS <- UniGASFit(gas_spec, returns_c, Compute.SE = FALSE)
fit_MSGARCH <- FitML(msgarch_spec, returns_c, ctr = list(do.se = FALSE))

vol_ins[1:ins, "GARCH"] <- sigma(fit_GARCH)^2
vol_ins[1:ins, "GAS"] <- fit_GAS@GASDyn$mTheta[2, 1:ins] * fit_GAS@GASDyn$mTheta[3, 1] / (fit_GAS@GASDyn$mTheta[3, 1] - 2)
vol_ins[1:ins, "MSGARCH"] <- Volatility(fit_MSGARCH)^2

# OoS
ES_1 <- ES_5 <- VaR_1 <- VaR_5 <- vol_oos <- matrix(0, ncol = 3, nrow = oos, dimnames = list(NULL, c("GARCH", "GAS", "MSGARCH")))
r_oos <- c()

for (i in 1:oos) {
  print(i)
  returns_window <- returns[i:(i + ins - 1)]
  mu <- mean(returns_window)
  #returns_window <- data[i:(i + ins - 1), "RETURNS_APPLE", drop = FALSE]
  #mu <- apply(returns_window, 2, mean)
  returns_c <- scale(returns_window, scale = FALSE)
  
  for (j in 1:ncol(returns_c)) {
    if(abs(acf(returns_c[, j], plot = FALSE)$acf[2]) > 2/sqrt(nrow(returns_c))) {
      ar_fit <- ar.yw(returns_c[, j], 3, aic = TRUE, se.fit = FALSE)
      returns_c[, j] <- ar_fit$resid
    }
  }
  
  fit_GARCH <- tryCatch(
    ugarchfit(garch_spec, returns_c, solver = "hybrid"),
    error = function(e) NULL
  )
  
  fit_GAS <- tryCatch(
    UniGASFit(gas_spec, returns_c, Compute.SE = FALSE),
    error = function(e) NULL
  )
  
  #fit_MSGARCH <- tryCatch(
  #  FitML(msgarch_spec, returns_c, ctr = list(do.se = FALSE)),
  #  error = function(e) NULL
  #)
  
  # one-step-ahead volatility^2
  vol_oos[i, "GARCH"] <- ugarchforecast(fit_GARCH, n.ahead = 1)@forecast$sigmaFor[1]^2
  #vol_oos[i, "MSGARCH"] <- predict(fit_MSGARCH, nahead = 1)$vol^2
  vol_oos[i, "GAS"] <- UniGASFor(fit_GAS, H = 1)@Forecast$PointForecast[, 2] * fit_GAS@GASDyn$mTheta[3, 1] /(fit_GAS@GASDyn$mTheta[3, 1] - 2)
  
  # adjusted values
  vol_ins[i + ins, "GARCH"] <- sigma(fit_GARCH)[ins]^2
  vol_ins[i + ins, "GAS"] <- fit_GAS@GASDyn$mTheta[2, ins] * fit_GAS@GASDyn$mTheta[3, 1] / (fit_GAS@GASDyn$mTheta[3, 1] - 2)
  #vol_ins[i + ins, "MSGARCH"] <- Volatility(fit_MSGARCH)[ins]^2
  
  # Residuals
  res_GARCH <- as.numeric(returns_c/sigma(fit_GARCH))
  res_GAS <- as.numeric(returns_c/sqrt(fit_GAS@GASDyn$mTheta[2, 1:ins] * fit_GAS@GASDyn$mTheta[3, 1] /(fit_GAS@GASDyn$mTheta[3, 1] - 2)))
  #res_MSGARCH <- as.numeric(returns_c/ Volatility(fit_MSGARCH))
  
  # VaR and ES 
  # 1%
  VaR_1[i, "GARCH"] = mu + sqrt(vol_oos[i, "GARCH"]) * quantile(res_GARCH, 0.01)
  VaR_1[i, "GAS"] = mu + sqrt(vol_oos[i, "GAS"] )* quantile(res_GAS, 0.01)
  #VaR_1[i, "MSGARCH"] = mu + sqrt(vol_oos[i, "MSGARCH"]) * quantile(res_MSGARCH, 0.01)
  
  ES_1[i, "GARCH"] <- mean(returns_window[returns_window < VaR_1[i, "GARCH"]], na.rm = TRUE)
  ES_1[i, "GAS"] <- mean(returns_window[returns_window < VaR_1[i, "GAS"]], na.rm = TRUE)
  #ES_1[i, "MSGARCH"] <- mean(returns_window[returns_window < VaR_1[i, "MSGARCH"]], na.rm = TRUE)
  
  # 5%
  VaR_5[i, "GARCH"] = mu + sqrt(vol_oos[i, "GARCH"]) * quantile(res_GARCH, 0.05)
  VaR_5[i, "GAS"] = mu + sqrt(vol_oos[i, "GAS"] )* quantile(res_GAS, 0.05)
  #VaR_5[i, "MSGARCH"] = mu + sqrt(vol_oos[i, "MSGARCH"]) * quantile(res_MSGARCH, 0.05)
  
  ES_5[i, "GARCH"] <- mean(returns_window[returns_window < VaR_5[i, "GARCH"]], na.rm = TRUE)
  ES_5[i, "GAS"] <- mean(returns_window[returns_window < VaR_5[i, "GAS"]], na.rm = TRUE)
  #ES_5[i, "MSGARCH"] <- mean(returns_window[returns_window < VaR_5[i, "MSGARCH"]], na.rm = TRUE)
  
  r_oos[i] <- returns[i + ins]
}

# ins data
df_ins <- data.frame(
  Date = data$DATE,
  Returns = data$RETURNS_APPLE,
  Sigma2_GARCH = vol_ins[, "GARCH"],
  Sigma2_GAS = vol_ins[, "GAS"],
  #Sigma2_MSGARCH = vol_ins[, "MSGARCH"]
)

# oos data
df_oos <- data.frame(
  Date = data$DATE[(ins + 1):(ins + oos)],
  Returns = r_oos,
  Vol_GARCH = vol_oos[, "GARCH"],
  Vol_GAS = vol_oos[, "GAS"],
  #Vol_MSGARCH = vol_oos[, "MSGARCH"],
  
  VaR_GARCH_1 = VaR_1[, "GARCH"],
  VaR_GAS_1 = VaR_1[, "GAS"],
  #VaR_MSGARCH_1 = VaR_1[, "MSGARCH"],
  
  ES_GARCH_1 = ES_1[, "GARCH"],
  ES_GAS_1 = ES_1[, "GAS"],
  #ES_MSGARCH_1 = ES_1[, "MSGARCH"],
  
  VaR_GARCH_5 = VaR_5[, "GARCH"],
  VaR_GAS_5 = VaR_5[, "GAS"],
  #VaR_MSGARCH_5 = VaR_5[, "MSGARCH"],
  
  ES_GARCH_5 = ES_5[, "GARCH"],
  ES_GAS_5 = ES_5[, "GAS"],
  #ES_MSGARCH_5 = ES_5[, "MSGARCH"]
)

sum(returns_window < VaR_1[i, "GARCH"])

#[1] 1500
# Error in UseMethod("ugarchforecast") : 
#  método não aplicável para 'ugarchforecast' aplicado a um objeto de classe "NULL"
