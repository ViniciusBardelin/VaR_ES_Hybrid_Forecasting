#############################################################################
#####                       Empirical Application                       #####
#############################################################################

library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(readxl)
library(readr)
library(xts)
library(tidyr)
library(highfrequency)
library(ggplot2)
library(data.table)
library(tidyverse) 

STOCK = 'AAPL' # (AAPL, MSFT)
file_path <- paste0(STOCK, "_DATA.xlsx")

df <- read_excel(file_path)
df$DATE <- as.Date(df$DATE)

returns <- df[[paste0("RETURNS_", STOCK)]]

df <- df[!apply(df, 1, function(r) any(as.character(r) == "-")), ]
sum(df == 0, na.rm = TRUE)
sum(returns == 0, na.rm = TRUE)
df <- df[!apply(df[sapply(df, is.numeric)] == 0, 1, any), ]

n_ins <- 2500
n_tot <- nrow(df)
n_oos <- n_tot - n_ins

# Specs
garch_spec <- ugarchspec(variance.model = list(model= "sGARCH", garchOrder = c(1,1)), mean.model = list(armaOrder  = c(0,0), include.mean = FALSE), distribution.model = "std")
msgarch_spec <- CreateSpec(variance.spec = list(model = "sGARCH"), distribution.spec = list(distribution = "std"), switch.spec = list(do.mix = FALSE, K = 2))
gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity", GASPar = list(scale = TRUE))

# InS
sigma2_completo <- matrix(NA_real_, nrow = n_tot, ncol = 3, dimnames = list(NULL, c("GARCH", "MSGARCH", "GAS")))
#sigma2_completo <- matrix(NA_real_, nrow = n_tot, ncol = 1, dimnames = list(NULL, c("GARCH")))

returns_c <- scale(returns[1:n_ins], scale = FALSE)

fit_GARCH <- ugarchfit(garch_spec, returns_c, solver = "hybrid")
fit_MSGARCH <- FitML(msgarch_spec, returns_c, ctr = list(do.se = FALSE))
fit_GAS <- UniGASFit(gas_spec, returns_c, Compute.SE = FALSE)


sigma2_completo[1:n_ins, "GARCH"] <- sigma(fit_GARCH)^2
sigma2_completo[1:n_ins, "MSGARCH"] <- Volatility(fit_MSGARCH)^2
sigma2_completo[1:n_ins, "GAS"] <- fit_GAS@GASDyn$mTheta[2, 1:n_ins] * fit_GAS@GASDyn$mTheta[3, 1] / (fit_GAS@GASDyn$mTheta[3, 1] - 2)

# OoS
ES_1 <- <- ES_2 <- ES_5 <- VaR_1 <- VaR_2 <- VaR_5 <- sigma2 <- matrix(0, ncol = 3, nrow = n_oos, dimnames = list(NULL, c("GARCH", "MSGARCH", "GAS")))
#ES_1 <- ES_2 <- ES_5 <- VaR_1 <- VaR_2 <- VaR_5 <- sigma2 <- matrix(0, ncol = 1, nrow = n_oos, dimnames = list(NULL, c("GARCH")))
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
  fit_MSGARCH <- FitML(msgarch_spec, returns_c, ctr = list(do.se = FALSE))
  fit_GAS <- UniGASFit(gas_spec, returns_c, Compute.SE = FALSE)
  
  sigma2[i, "GARCH"] <- ugarchforecast(fit_GARCH, n.ahead = 1)@forecast$sigmaFor[1]^2
  sigma2[i, "MSGARCH"] <- predict(fit_MSGARCH , nahead = 1)$vol^2
  sigma2[i, "GAS"] <- UniGASFor(fit_GAS, H = 1)@Forecast$PointForecast[, 2] * fit_GAS@GASDyn$mTheta[3, 1] / (fit_GAS@GASDyn$mTheta[3, 1] - 2)
  
  sigma2_completo[i + n_ins, "GARCH"] <- sigma(fit_GARCH)[n_ins]^2
  sigma2_completo[i + n_ins, "MSGARCH"] <- Volatility(fit_MSGARCH)[n_ins]^2
  sigma2_completo[i + n_ins, "GAS"] <- fit_GAS@GASDyn$mTheta[2, n_ins] * fit_GAS@GASDyn$mTheta[3, 1] / (fit_GAS@GASDyn$mTheta[3, 1] - 2)
  
  res_GARCH <- as.numeric(returns_c/sigma(fit_GARCH))
  res_MSGARCH <- as.numeric(returns_c/Volatility(fit_MSGARCH))
  res_GAS <- as.numeric(returns_c/sqrt(fit_GAS@GASDyn$mTheta[2, 1:n_ins] * fit_GAS@GASDyn$mTheta[3, 1] / (fit_GAS@GASDyn$mTheta[3, 1] - 2)))
  
  # 1%
  VaR_1[i, "GARCH"] = mu + sqrt(sigma2[i, "GARCH"]) * quantile(res_GARCH, 0.01)
  VaR_1[i, "MSGARCH"] = mu + sqrt(sigma2[i, "MSGARCH"]) * quantile(res_MSGARCH, 0.01)
  VaR_1[i, "GAS"] = mu + sqrt(sigma2[i, "GAS"] )* quantile(res_GAS, 0.01)
  
  ES_1[i, "GARCH"] <- mean(returns_window[returns_window < VaR_1[i, "GARCH"]])
  ES_1[i, "GAS"] <- mean(returns_window[returns_window < VaR_1[i, "GAS"]])
  ES_1[i, "MSGARCH"] <- mean(returns_window[returns_window < VaR_1[i, "MSGARCH"]])
  
  # 2.5%
  VaR_2[i, "GARCH"] = mu + sqrt(sigma2[i, "GARCH"]) * quantile(res_GARCH, 0.025)
  VaR_2[i, "MSGARCH"] = mu + sqrt(sigma2[i, "MSGARCH"]) * quantile(res_MSGARCH, 0.025)
  VaR_2[i, "GAS"] = mu + sqrt(sigma2[i, "GAS"] )* quantile(res_GAS, 0.025)

  ES_2[i, "GARCH"] <- mean(returns_window[returns_window < VaR_2[i, "GARCH"]])
  ES_2[i, "GAS"] <- mean(returns_window[returns_window < VaR_2[i, "GAS"]])
  ES_2[i, "MSGARCH"] <- mean(returns_window[returns_window < VaR_2[i, "MSGARCH"]])
  
  # 5%
  VaR_5[i, "GARCH"] = mu + sqrt(sigma2[i, "GARCH"]) * quantile(res_GARCH, 0.05)
  VaR_5[i, "MSGARCH"] = mu + sqrt(sigma2[i, "MSGARCH"]) * quantile(res_MSGARCH, 0.05)
  VaR_5[i, "GAS"] = mu + sqrt(sigma2[i, "GAS"] )* quantile(res_GAS, 0.05)
  
  ES_5[i, "GARCH"] <- mean(returns_window[returns_window < VaR_5[i, "GARCH"]])
  ES_5[i, "GAS"] <- mean(returns_window[returns_window < VaR_5[i, "GAS"]])
  ES_5[i, "MSGARCH"] <- mean(returns_window[returns_window < VaR_5[i, "MSGARCH"]])
  
  r_oos[i] <- returns[i + n_ins]
  
}

# InS
df_sigma2_completo <- data.frame(
  Date = df$DATE,
  Returns = df[[paste0("RETURNS_", STOCK)]],
  Sigma2_GARCH = sigma2_completo[, "GARCH"],
  #Sigma2_MSGARCH = sigma2_completo[, "MSGARCH"],
  #Sigma2_GAS = sigma2_completo[, "GAS"],
  RV_STOCK = df[[paste0("RV_", STOCK)]]
)
colnames(df_sigma2_completo)[ncol(df_sigma2_completo)] <- paste0("RV_", STOCK)


write.csv(df_sigma2_completo, "AAPL_GARCH_adjusted_values.csv", row.names = FALSE)

# OoS
df_oos <- data.frame(
  Date = df$DATE[(n_ins + 1):n_tot],
  Return = r_oos,
  Vol_GARCH = sqrt(sigma2[, "GARCH"]),
  #Vol_MSGARCH = sqrt(sigma2[, "MSGARCH"]),
  #Vol_GAS = sqrt(sigma2[, "GAS"]),
  
  VaR_GARCH_1 = VaR_1[, "GARCH"],
  #VaR_MSGARCH_1 = VaR_1[, "MSGARCH"],
  #VaR_GAS_1 = VaR_1[, "GAS"],
  ES_GARCH_1 = ES_1[, "GARCH"],
  #ES_MSGARCH_1 = ES_1[, "MSGARCH"],
  #ES_GAS_1 = ES_1[, "GAS"],
  
  VaR_GARCH_2 = VaR_2[, "GARCH"],
  #VaR_MSGARCH_2 = VaR_2[, "MSGARCH"],
  #VaR_GAS_2 = VaR_2[, "GAS"],
  ES_GARCH_2 = ES_2[, "GARCH"],
  #ES_MSGARCH_2 = ES_2[, "MSGARCH"],
  #ES_GAS_2 = ES_2[, "GAS"],
  
  VaR_GARCH_5 = VaR_5[, "GARCH"],
  #VaR_MSGARCH_5 = VaR_5[, "MSGARCH"],
  #VaR_GAS_5 = VaR_5[, "GAS"],
  ES_GARCH_5 = ES_5[, "GARCH"],
  #ES_MSGARCH_5 = ES_5[, "MSGARCH"],
  #ES_GAS_5 = ES_5[, "GAS"],
  
  RV_TEMP = df[[paste0("RV_", STOCK)]][(n_ins + 1):n_tot]
)
colnames(df_oos)[colnames(df_oos) == "RV_TEMP"] <- paste0("RV_", STOCK)

write.csv(df_oos, "APPL_GARCH_oos_data.csv", row.names = FALSE)

# Check VaR 1%
sum(df_oos$Return < df_oos$VaR_GARCH_1)/nrow(df_oos)
sum(df_oos$Return < df_oos$VaR_MSGARCH_1)/nrow(df_oos) 
sum(df_oos$Return < df_oos$VaR_GAS_1)/nrow(df_oos) 

# Check VaR 5%
sum(df_oos$Return < df_oos$VaR_GARCH_5)/nrow(df_oos) 
sum(df_oos$Return < df_oos$VaR_MSGARCH_5)/nrow(df_oos) 
sum(df_oos$Return < df_oos$VaR_GAS_5)/nrow(df_oos) 

###########
#   HAR   #
###########

# InS
RV <- as.xts(df[[paste0("RV_", STOCK)]], order.by = df$DATE)
RV_ins <- RV[1:n_ins]
sigmaHAR_completo <- matrix(NA_real_, nrow = n_tot, ncol = 1, dimnames = list(NULL, c("HAR")))
fit_HAR <- HARmodel(RV_ins, periods = c(1,5,22), RVest = c("rCov"), type = "HAR", h = 1, transform = NULL, inputType = "RM")
sigmaHAR_completo[23:n_ins, "HAR"] <- fit_HAR$fitted.values

# OoS
ES_1 <- ES_2 <- ES_5 <- VaR_1 <- VaR_2 <- VaR_5 <- sigmaHAR <- matrix(0, ncol = 1, nrow = n_oos, dimnames = list(NULL, c("HAR")))
r_oos <- c()
for (i in 1:n_oos) {
  print(i)
  
  returns_window <- returns[i:(i + n_ins - 1)]
  mu <- mean(returns_window)
  returns_c <- scale(returns_window, scale = FALSE)
  
  rv_window <- as.xts(RV[i:(i + n_ins - 1)])
  
  fit_HAR <- HARmodel(
    rv_window,
    periods = c(1, 5, 22),
    RVest = c("rCov"),
    type = "HAR",
    h = 1,
    inputType = "RM"
  )
  
  har_forecast <- as.numeric(predict(fit_HAR))
  
  sigma2_forecast <- har_forecast
  
  sigmaHAR[i, "HAR"] <- sigma2_forecast
  sigmaHAR_completo[i + n_ins, "HAR"] <- sigma2_forecast
  
  rv_hat_is <- as.numeric(na.omit(fit_HAR$fitted.values))

  k <- length(rv_hat_is)
  r_c_is <- tail(returns_c, k)
  
  sigma_hat_is <- sqrt(rv_hat_is)
  res_HAR <- r_c_is / sigma_hat_is
  
  q01 <- quantile(res_HAR, 0.01, na.rm = TRUE)
  q02 <- quantile(res_HAR, 0.025, na.rm = TRUE)
  q05 <- quantile(res_HAR, 0.05, na.rm = TRUE)
  
  #es_z01 <- mean(res_HAR[res_HAR < q01], na.rm = TRUE)
  #es_z05 <- mean(res_HAR[res_HAR < q05], na.rm = TRUE)
  
  sigma_oos <- sqrt(sigma2_forecast)
  
  VaR_1[i, "HAR"] <- mu + sigma_oos * q01
  VaR_2[i, "HAR"] <- mu + sigma_oos * q02
  VaR_5[i, "HAR"] <- mu + sigma_oos * q05
  
  #ES_1[i, "HAR"]  <- mu + sigma_oos * es_z01
  #ES_5[i, "HAR"]  <- mu + sigma_oos * es_z05
  
  ES_1[i, "HAR"]  <- mean(returns_window[returns_window < VaR_1[i, "HAR"]], na.rm = TRUE)
  ES_2[i, "HAR"]  <- mean(returns_window[returns_window < VaR_2[i, "HAR"]], na.rm = TRUE)
  ES_5[i, "HAR"]  <- mean(returns_window[returns_window < VaR_5[i, "HAR"]], na.rm = TRUE)
  
  r_oos[i] <- returns[i + n_ins]
}


# InS
df_sigmaHAR_completo <- data.frame(
  Date           = df$DATE,
  Returns        = df[[paste0("RETURNS_", STOCK)]],
  Sigma2_HAR     = sigmaHAR_completo[, "HAR"],
  RV_TEMP        = df[[paste0("RV_", STOCK)]]
)
colnames(df_sigmaHAR_completo)[colnames(df_sigmaHAR_completo) == "RV_TEMP"] <- paste0("RV_", STOCK)

write.csv(df_sigmaHAR_completo, "AAPL_HAR_adjusted_values_2_5.csv", row.names = FALSE)

# OoS
df_oos_HAR <- data.frame(
  Date      = df$DATE[(n_ins + 1):n_tot],
  Return    = r_oos,
  Vol_HAR   = sqrt(sigmaHAR[, "HAR"]),
  
  VaR_HAR_1 = VaR_1[, "HAR"],
  ES_HAR_1  = ES_1[, "HAR"],
 
  VaR_HAR_2 = VaR_2[, "HAR"],
  ES_HAR_2  = ES_2[, "HAR"],
  
  VaR_HAR_5 = VaR_5[, "HAR"],
  ES_HAR_5  = ES_5[, "HAR"],
  
  RV_TEMP   = df[[paste0("RV_", STOCK)]][(n_ins + 1):n_tot]
)
colnames(df_oos_HAR)[ncol(df_oos_HAR)] <- paste0("RV_", STOCK)

write.csv(df_oos_HAR, "AAPL_HAR_oos_data_2_5.csv", row.names = FALSE)

# Check VaR
sum(df_oos_HAR$Return < df_oos_HAR$VaR_HAR_1)/nrow(df_oos_HAR)
sum(df_oos_HAR$Return < df_oos_HAR$VaR_HAR_2)/nrow(df_oos_HAR) 
sum(df_oos_HAR$Return < df_oos_HAR$VaR_HAR_5)/nrow(df_oos_HAR) 

plot(df_oos_HAR$Date, df_oos_HAR$Return, type = 'l')
lines(df_oos_HAR$Date, df_oos_HAR$VaR_HAR_1, type = 'l', col = 'red')

plot(df_oos_HAR$Date, df_oos_HAR$Return, type = 'l')
lines(df_oos_HAR$Date, df_oos_HAR$VaR_HAR_5, type = 'l', col = 'red')
