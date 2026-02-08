#############################################################################
#####                       Empirical Application                       #####
#############################################################################

library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(readxl)
library(xts)
library(tidyr)
library(highfrequency)

df <- read_excel("APPLE_DATA.xlsx")
df$DATE <- as.Date(df$DATE)
returns <- df$RETURNS_APPLE

n_ins <- 2500
n_tot <- length(returns)
n_oos <- n_tot - n_ins

# Specs
garch_spec<- ugarchspec(variance.model = list(model= "sGARCH", garchOrder = c(1,1)), mean.model = list(armaOrder  = c(0,0), include.mean = FALSE), distribution.model = "std")
msgarch_spec <- CreateSpec(variance.spec = list(model = "sGARCH"), distribution.spec = list(distribution = "std"), switch.spec = list(do.mix = FALSE, K = 2))
gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity", GASPar = list(scale = TRUE))

# InS
sigma2_completo <- matrix(NA_real_, nrow = n_tot, ncol = 3,
                          dimnames = list(NULL, c("GARCH", "MSGARCH", "GAS")))

returns_c <- scale(returns[1:n_ins], scale = FALSE)

fit_GARCH <- ugarchfit(garch_spec, returns_c, solver = "hybrid")
fit_GAS <- UniGASFit(gas_spec, returns_c, Compute.SE = FALSE)
fit_MSGARCH <- FitML(msgarch_spec, returns_c, ctr = list(do.se = FALSE))

sigma2_completo[1:n_ins, "GARCH"] <- sigma(fit_GARCH)^2
sigma2_completo[1:n_ins, "GAS"] <- fit_GAS@GASDyn$mTheta[2, 1:n_ins] * fit_GAS@GASDyn$mTheta[3, 1] / (fit_GAS@GASDyn$mTheta[3, 1] - 2)
sigma2_completo[1:n_ins, "MSGARCH"] <- Volatility(fit_MSGARCH)^2

# OoS
ES_1 <- ES_5 <- VaR_1 <- VaR_5 <- sigma2 <- matrix(0, ncol = 3, nrow = n_oos, dimnames = list(NULL, c("GARCH", "MSGARCH", "GAS")))
r_oos <- c()
for (i in 1:n_oos) {
  print(i)
  returns_window <- returns[i:(i + n_ins - 1)]
  mu <- mean(returns_window)
  returns_c <- scale(returns_window, scale = FALSE)
  
  for (j in 1:ncol(returns_c)) {
    
    acf1 <- tryCatch(
      acf(returns_c[, j], plot = FALSE)$acf[2],
      error = function(e) NA_real_
    )
    
    if (!is.na(acf1) && abs(acf1) > 2/sqrt(nrow(returns_c))) {
      
      ar_fit <- tryCatch(
        ar.yw(returns_c[, j], order.max = 3, aic = TRUE, se.fit = FALSE),
        error = function(e) NULL
      )
      
      if (!is.null(ar_fit)) {
        r <- as.numeric(ar_fit$resid)
        
        
        if (anyNA(r)) {
          idx <- is.na(r)
          r[idx] <- returns_c[idx, j]   
        }
        
        if (all(is.finite(r)) && sd(r) > 1e-10) {
          returns_c[, j] <- r
        }
      }
    }
  }
  
  fit_GARCH <- ugarchfit(garch_spec, returns_c, solver = "hybrid")
  fit_GAS <- UniGASFit(gas_spec, returns_c, Compute.SE = FALSE)
  fit_MSGARCH <- FitML(msgarch_spec, returns_c, ctr = list(do.se = FALSE))
  
  sigma2[i, "GARCH"] <- ugarchforecast(fit_GARCH, n.ahead = 1)@forecast$sigmaFor[1]^2
  sigma2[i, "MSGARCH"] <- predict(fit_MSGARCH , nahead = 1)$vol^2
  sigma2[i, "GAS"] <- UniGASFor(fit_GAS, H = 1)@Forecast$PointForecast[, 2] * fit_GAS@GASDyn$mTheta[3, 1] /(fit_GAS@GASDyn$mTheta[3, 1] - 2)
  
  sigma2_completo[i + n_ins, "GARCH"] <- sigma(fit_GARCH)[n_ins]^2
  sigma2_completo[i + n_ins, "GAS"] <- fit_GAS@GASDyn$mTheta[2, n_ins] * fit_GAS@GASDyn$mTheta[3, 1] / (fit_GAS@GASDyn$mTheta[3, 1] - 2)
  sigma2_completo[i + n_ins, "MSGARCH"] <- Volatility(fit_MSGARCH)[n_ins]^2
  
  res_GARCH <- as.numeric(returns_c/sigma(fit_GARCH))
  res_GAS <-   as.numeric(returns_c/sqrt(fit_GAS@GASDyn$mTheta[2, 1:n_ins] * fit_GAS@GASDyn$mTheta[3, 1] /(fit_GAS@GASDyn$mTheta[3, 1] - 2)))
  res_MSGARCH <- as.numeric(returns_c/ Volatility(fit_MSGARCH))
  
  VaR_1[i, "GARCH"] = mu + sqrt(sigma2[i, "GARCH"]) * quantile(res_GARCH, 0.01)
  VaR_1[i, "GAS"] = mu + sqrt(sigma2[i, "GAS"] )* quantile(res_GAS, 0.01)
  VaR_1[i, "MSGARCH"] = mu + sqrt(sigma2[i, "MSGARCH"]) * quantile(res_MSGARCH, 0.01)
  
  ES_1[i, "GARCH"] <- mean(returns_window[returns_window < VaR_1[i, "GARCH"]])
  ES_1[i, "GAS"] <- mean(returns_window[returns_window < VaR_1[i, "GAS"]])
  ES_1[i, "MSGARCH"] <- mean(returns_window[returns_window < VaR_1[i, "MSGARCH"]])
  
  VaR_5[i, "GARCH"] = mu + sqrt(sigma2[i, "GARCH"]) * quantile(res_GARCH, 0.05)
  VaR_5[i, "GAS"] = mu + sqrt(sigma2[i, "GAS"] )* quantile(res_GAS, 0.05)
  VaR_5[i, "MSGARCH"] = mu + sqrt(sigma2[i, "MSGARCH"]) * quantile(res_MSGARCH, 0.05)
  
  ES_5[i, "GARCH"] <- mean(returns_window[returns_window < VaR_5[i, "GARCH"]])
  ES_5[i, "GAS"] <- mean(returns_window[returns_window < VaR_5[i, "GAS"]])
  ES_5[i, "MSGARCH"] <- mean(returns_window[returns_window < VaR_5[i, "MSGARCH"]])
  
  r_oos[i] <- returns[i + n_ins]
  
}

# InS
df_sigma2_completo <- data.frame(
  Date = df$DATE,
  Returns = df$RETURNS_APPLE,
  Sigma2_GARCH = sigma2_completo[, "GARCH"],
  Sigma2_GAS = sigma2_completo[, "GAS"],
  Sigma2_MSGARCH = sigma2_completo[, "MSGARCH"],
  RV_APPLE = df$RV_APPLE
)

write.csv(df_sigma2_completo, "ins_data.csv", row.names = FALSE)

# OoS
df_oos <- data.frame(
  Date = df$DATE[(n_ins + 1):n_tot],
  Return = r_oos,
  Vol_GARCH = sqrt(sigma2[, "GARCH"]),
  Vol_MSGARCH = sqrt(sigma2[, "MSGARCH"]),
  Vol_GAS = sqrt(sigma2[, "GAS"]),
  
  VaR_GARCH_1 = VaR_1[, "GARCH"],
  VaR_MSGARCH_1 = VaR_1[, "MSGARCH"],
  VaR_GAS_1 = VaR_1[, "GAS"],
  ES_GARCH_1 = ES_1[, "GARCH"],
  ES_MSGARCH_1 = ES_1[, "MSGARCH"],
  ES_GAS_1 = ES_1[, "GAS"],
  
  VaR_GARCH_5 = VaR_5[, "GARCH"],
  VaR_MSGARCH_5 = VaR_5[, "MSGARCH"],
  VaR_GAS_5 = VaR_5[, "GAS"],
  ES_GARCH_5 = ES_5[, "GARCH"],
  ES_MSGARCH_5 = ES_5[, "MSGARCH"],
  ES_GAS_5 = ES_5[, "GAS"],
  
  RV_APPLE = df$RV_APPLE[(n_ins + 1):n_tot]
)

write.csv(df_oos, "oos_data.csv", row.names = FALSE)

# Check VaR 1%
sum(df_oos$Return < df_oos$VaR_GARCH_1)/2846 # [1] 0.01124385
sum(df_oos$Return < df_oos$VaR_MSGARCH_1)/2846 # [1] 0.01089248
sum(df_oos$Return < df_oos$VaR_GAS_1)/2846 # [1] 0.00983837

# Check VaR 5%
sum(df_oos$Return < df_oos$VaR_GARCH_5)/2846 # [1] 0.05165144
sum(df_oos$Return < df_oos$VaR_MSGARCH_5)/2846 # [1] 0.05411103
sum(df_oos$Return < df_oos$VaR_GAS_5)/2846 # [1] 0.04884048

###########
#   HAR   #
###########

# InS
RV <- as.xts(df$RV_APPLE, order.by = df$DATE)

RV_ins <- RV[1:2500]

sigmaHAR_completo <- matrix(NA_real_, nrow = n_tot, ncol = 1,
                            dimnames = list(NULL, c("HAR")))

fit_HAR <- HARmodel(RV_ins, periods = c(1,5,22), RVest = c("rCov"), 
                    type = "HAR", h = 1, transform = NULL, inputType = "RM")

sigmaHAR_completo[23:n_ins, "HAR"] <- fit_HAR$fitted.values

# OoS
ES_1 <- ES_5 <- VaR_1 <- VaR_5 <- sigmaHAR <- matrix(0, ncol = 1, nrow = n_oos, dimnames = list(NULL, c("HAR")))
r_oos <- c()
for (i in 1:n_oos) {
  print(i)
  
  returns_window <- returns[i:(i + n_ins - 1)]
  mu <- mean(returns_window)
  
  returns_c <- as.numeric(returns_window - mu)
  
  rv_window <- as.xts(RV[i:(i + n_ins - 1)])
  
  fit_HAR <- HARmodel(
    rv_window,
    periods = c(1, 5, 22),
    RVest = c("rCov"),
    type = "HAR",
    h = 1,
    transform = NULL,
    inputType = "RM"
  )
  
  sigmaHAR[i, "HAR"] <- predict(fit_HAR)
  
  sigmaHAR_completo[i + n_ins, "HAR"] <- as.numeric(tail(fit_HAR$fitted.values, 1))
  
  rv_hat_is <- as.numeric(na.omit(fit_HAR$fitted.values))  
  r_c_is    <- returns_c[23:n_ins]                         
  
  eps <- 1e-12
  sigma_hat_is <- sqrt(pmax(rv_hat_is, eps))               
  
  res_HAR <- r_c_is / sigma_hat_is                    
  
  q01 <- quantile(res_HAR, 0.01, na.rm = TRUE)
  q05 <- quantile(res_HAR, 0.05, na.rm = TRUE)
  
  sigma_oos <- sqrt(as.numeric(sigmaHAR[i, "HAR"]))        
  
  VaR_1[i, "HAR"] <- mu + sigma_oos * q01
  ES_1[i, "HAR"]  <- mean(returns_window[returns_window < VaR_1[i, "HAR"]], na.rm = TRUE)
  
  VaR_5[i, "HAR"] <- mu + sigma_oos * q05
  ES_5[i, "HAR"]  <- mean(returns_window[returns_window < VaR_5[i, "HAR"]], na.rm = TRUE)
  
  r_oos[i] <- returns[i + n_ins]
}

# InS
df_sigmaHAR_completo <- data.frame(
  Date = df$DATE,
  Returns = df$RETURNS_APPLE,
  Sigma2_GARCH = sigmaHAR_completo[, "HAR"],
  RV_APPLE = df$RV_APPLE
)

write.csv(df_sigmaHAR_completo, "ins_HAR_data.csv", row.names = FALSE)

# OoS
df_oos_HAR <- data.frame(
  Date = df$DATE[(n_ins + 1):n_tot],
  Return = r_oos,
  Vol_HAR = sqrt(sigmaHAR[, "HAR"]),
  
  VaR_HAR_1 = VaR_1[, "HAR"],
  ES_HAR_1 = ES_1[, "HAR"],
  
  VaR_HAR_5 = VaR_5[, "HAR"],
  ES_HAR_5 = ES_5[, "HAR"],
  
  RV_APPLE = df$RV_APPLE[(n_ins + 1):n_tot]
)

write.csv(df_oos_HAR, "oos_HAR_data.csv", row.names = FALSE)

# Check VaR
sum(df_oos_HAR$Return < df_oos_HAR$VaR_HAR_1)/2846 # 0.01405481
sum(df_oos_HAR$Return < df_oos_HAR$VaR_HAR_5)/2846 # 0.05551651
