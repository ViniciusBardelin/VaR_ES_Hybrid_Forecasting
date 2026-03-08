library(dplyr)
library(readr)
library(GAS)
library(lubridate)
library(esback)
source("Function_VaR_VQR.R")
source("Optimizations.R")

# Univariate models backtesting
df_uni_raw <- read_csv("AMZN_oos_data.csv", show_col_types = FALSE) %>%
  mutate(Date = ymd(Date))

df_har_std <- read_csv("AMZN_oos_HAR_data.csv", show_col_types = FALSE) %>%
  mutate(Date = ymd(Date)) %>%
  transmute(
    Date,
    Vol_HAR   = Vol_HAR,
    VaR_HAR_1 = VaR_HAR_1,
    ES_HAR_1  = ES_HAR_1,
    VaR_HAR_5 = VaR_HAR_5,
    ES_HAR_5  = ES_HAR_5
  )

df_all_raw <- df_uni_raw %>%
  mutate(Date = ymd(Date)) %>%
  left_join(df_har_std, by = "Date")

run_univar_backtest <- function(df, model = c("GARCH", "MSGARCH", "GAS", "HAR")) {
  model <- match.arg(model)
  
  vol_col   <- paste0("Vol_", model)
  var1_col  <- paste0("VaR_", model, "_1")
  es1_col   <- paste0("ES_",  model, "_1")
  var5_col  <- paste0("VaR_", model, "_5")
  es5_col   <- paste0("ES_",  model, "_5")
  
  df_m <- df %>%
    transmute(
      Date,
      Return,
      Vol = .data[[vol_col]],
      VaR_1 = .data[[var1_col]],
      ES_1  = .data[[es1_col]],
      VaR_5 = .data[[var5_col]],
      ES_5  = .data[[es5_col]],
      Exceed_VaR_1 = Return < VaR_1,
      Exceed_VaR_5 = Return < VaR_5
    ) %>%
    filter(
      !is.na(Return),
      !is.na(Vol), Vol > 0,
      !is.na(VaR_1), !is.na(ES_1),
      !is.na(VaR_5), !is.na(ES_5)
    )
  
  # VaR backtests
  Back_VaR_1 <- BacktestVaR(df_m$Return, df_m$VaR_1, 0.01)
  Back_VaR_5 <- BacktestVaR(df_m$Return, df_m$VaR_5, 0.05)
  
  pVQR_1 <- VaR_VQR(df_m$Return, df_m$VaR_1, 0.01)
  pVQR_5 <- VaR_VQR(df_m$Return, df_m$VaR_5, 0.05)
  
  # ES backtests
  Back_ES_CoC_1 <- cc_backtest(df_m$Return, df_m$VaR_1, df_m$ES_1, df_m$Vol, 0.01)
  Back_ES_CoC_5 <- cc_backtest(df_m$Return, df_m$VaR_5, df_m$ES_5, df_m$Vol, 0.05)
  
  Back_ES_ER_1  <- er_backtest(df_m$Return, df_m$VaR_1, df_m$ES_1, df_m$Vol)
  Back_ES_ER_5  <- er_backtest(df_m$Return, df_m$VaR_5, df_m$ES_5, df_m$Vol)
  
  Back_ES_ESR_1 <- esr_backtest(df_m$Return, df_m$VaR_1, df_m$ES_1, alpha = 0.01, version = 1, B = 0)
  Back_ES_ESR_5 <- esr_backtest(df_m$Return, df_m$VaR_5, df_m$ES_5, alpha = 0.05, version = 1, B = 0)
  
  # p-values table
  df_pvals <- data.frame(
    Modelo = model,
    Nivel  = c("1%", "5%"),
    UC  = c(Back_VaR_1$LRuc["Pvalue"], Back_VaR_5$LRuc["Pvalue"]),
    CC  = c(Back_VaR_1$LRcc["Pvalue"], Back_VaR_5$LRcc["Pvalue"]),
    DQ  = c(Back_VaR_1$DQ$pvalue,      Back_VaR_5$DQ$pvalue),
    VQR = c(pVQR_1, pVQR_5),
    CoC = c(Back_ES_CoC_1$pvalue_twosided_general, Back_ES_CoC_5$pvalue_twosided_general),
    ER  = c(Back_ES_ER_1$pvalue_twosided_standardized, Back_ES_ER_5$pvalue_twosided_standardized),
    ESR = c(Back_ES_ESR_1$pvalue_twosided_asymptotic,  Back_ES_ESR_5$pvalue_twosided_asymptotic)
  )
  df_pvals_fmt <- df_pvals
  df_pvals_fmt[ , -(1:2)] <- lapply(df_pvals_fmt[ , -(1:2)], \(x) signif(as.numeric(x), 4))
  
  # Score functions
  Back_VaR_QL_1 <- Back_VaR_1$Loss$Loss
  Back_VaR_QL_5 <- Back_VaR_5$Loss$Loss
  
  Back_VaR_FZ_1 <- mean(FZLoss(df_m$Return, df_m$VaR_1, df_m$ES_1, 0.01))
  Back_VaR_FZ_5 <- mean(FZLoss(df_m$Return, df_m$VaR_5, df_m$ES_5, 0.05))
  
  Back_VaR_NZ_1 <- mean(NZ_deprecated(df_m$VaR_1, df_m$ES_1, df_m$Return, 0.01))
  Back_VaR_NZ_5 <- mean(NZ_deprecated(df_m$VaR_5, df_m$ES_5, df_m$Return, 0.05))
  
  Back_VaR_AL_1 <- mean(AL_deprecated(df_m$VaR_1, df_m$ES_1, df_m$Return, 0.01))
  Back_VaR_AL_5 <- mean(AL_deprecated(df_m$VaR_5, df_m$ES_5, df_m$Return, 0.05))
  
  df_scores <- data.frame(
    Modelo = model,
    Nivel  = c("1%", "5%"),
    QL = c(Back_VaR_QL_1, Back_VaR_QL_5),
    FZ = c(Back_VaR_FZ_1, Back_VaR_FZ_5),
    NZ = c(Back_VaR_NZ_1, Back_VaR_NZ_5),
    AL = c(Back_VaR_AL_1, Back_VaR_AL_5)
  )
  
  list(
    data_used = df_m,
    pvals = df_pvals_fmt,
    scores = df_scores
  )
}

models <- c("GARCH", "MSGARCH", "GAS", "HAR")

res_list <- lapply(models, \(m) run_univar_backtest(df_all_raw, m))
names(res_list) <- models

df_pvals_all  <- bind_rows(lapply(res_list, \(x) x$pvals))
df_scores_all <- bind_rows(lapply(res_list, \(x) x$scores))

options(digits = 5)
print(df_pvals_all)
print(df_scores_all)

viol_rates <- df_all_raw %>%
  summarise(
    viol_HAR_1 = 100 * mean(Return < VaR_HAR_1, na.rm = TRUE),
    viol_GARCH_1 = 100 * mean(Return < VaR_GARCH_1,   na.rm = TRUE),
    viol_GAS_1 = 100 * mean(Return < VaR_GAS_1,     na.rm = TRUE),
    viol_MSGARCH_1 = 100 * mean(Return < VaR_MSGARCH_1, na.rm = TRUE),
    viol_HAR_5 = 100 * mean(Return < VaR_HAR_5, na.rm = TRUE),
    viol_GARCH_5 = 100 * mean(Return < VaR_GARCH_5,   na.rm = TRUE),
    viol_GAS_5 = 100 * mean(Return < VaR_GAS_5,     na.rm = TRUE),
    viol_MSGARCH_5 = 100 * mean(Return < VaR_MSGARCH_5, na.rm = TRUE)
  )

print(viol_rates)

# Hybrid backtesting
df_lstm <- read_csv("AMZN_MSGARCH_LSTM_VaR_ES_1.csv", show_col_types = FALSE) %>%
  mutate(
    Date = ymd(Date),
    Vol2_LSTM = RV_hat,                 
    Vol_LSTM  = sqrt(pmax(Vol2_LSTM, 1e-12)),
    VaR_LSTM_1 = VaR_1,
    ES_LSTM_1  = ES_1,
    VaR_LSTM_5 = VaR_5,
    ES_LSTM_5  = ES_5,
    Exceed_VaR_LSTM_1 = Return < VaR_LSTM_1,
    Exceed_VaR_LSTM_5 = Return < VaR_LSTM_5
  ) %>%
  select(
    Date, Return, Vol2_LSTM, Vol_LSTM,
    VaR_LSTM_1, ES_LSTM_1,
    VaR_LSTM_5, ES_LSTM_5,
    Exceed_VaR_LSTM_1, Exceed_VaR_LSTM_5
  ) %>%
  filter(
    !is.na(Return),
    !is.na(VaR_LSTM_1), !is.na(ES_LSTM_1),
    !is.na(VaR_LSTM_5), !is.na(ES_LSTM_5),
    !is.na(Vol2_LSTM),
    Vol2_LSTM > 0
  )

100 * mean(df_lstm$Return < df_lstm$VaR_LSTM_1)  
100 * mean(df_lstm$Return < df_lstm$VaR_LSTM_5) 

# VaR Backtests
Back_VaR_LSTM_1 <- BacktestVaR(df_lstm$Return, df_lstm$VaR_LSTM_1, 0.01)
Back_VaR_LSTM_5 <- BacktestVaR(df_lstm$Return, df_lstm$VaR_LSTM_5, 0.05)

pVQR_1 <- VaR_VQR(df_lstm$Return, df_lstm$VaR_LSTM_1, 0.01)
pVQR_5 <- VaR_VQR(df_lstm$Return, df_lstm$VaR_LSTM_5, 0.05)

# ES Backtests
vol_lstm <- df_lstm$Vol_LSTM

Back_ES_CoC_LSTM_1 <- cc_backtest(df_lstm$Return, df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, vol_lstm, 0.01)
Back_ES_CoC_LSTM_5 <- cc_backtest(df_lstm$Return, df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, vol_lstm, 0.05)

Back_ES_ER_LSTM_1 <- er_backtest(df_lstm$Return, df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, vol_lstm)
Back_ES_ER_LSTM_5 <- er_backtest(df_lstm$Return, df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, vol_lstm)

Back_ES_ESR_LSTM_1_V1 <- esr_backtest(df_lstm$Return, df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, alpha = 0.01, version = 1, B = 0)
Back_ES_ESR_LSTM_5_V1 <- esr_backtest(df_lstm$Return, df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, alpha = 0.05, version = 1, B = 0)

# p-values table
df_pvals_tests <- data.frame(
  Nivel = c("1%", "5%"),
  
  UC  = c(Back_VaR_LSTM_1$LRuc["Pvalue"], Back_VaR_LSTM_5$LRuc["Pvalue"]),
  CC  = c(Back_VaR_LSTM_1$LRcc["Pvalue"], Back_VaR_LSTM_5$LRcc["Pvalue"]),
  DQ  = c(Back_VaR_LSTM_1$DQ$pvalue, Back_VaR_LSTM_5$DQ$pvalue),
  VQR = c(pVQR_1, pVQR_5),
  
  CoC = c(Back_ES_CoC_LSTM_1$pvalue_twosided_general,
          Back_ES_CoC_LSTM_5$pvalue_twosided_general),
  
  ER  = c(Back_ES_ER_LSTM_1$pvalue_twosided_standardized,
          Back_ES_ER_LSTM_5$pvalue_twosided_standardized),
  
  ESR = c(Back_ES_ESR_LSTM_1_V1$pvalue_twosided_asymptotic,
          Back_ES_ESR_LSTM_5_V1$pvalue_twosided_asymptotic)
)

df_pvals_tests_fmt <- df_pvals_tests
df_pvals_tests_fmt[,-1] <- lapply(df_pvals_tests_fmt[,-1], function(x) signif(as.numeric(x), 4))

options(digits = 5)
print(df_pvals_tests_fmt)

# Score functions
Back_VaR_QL_LSTM_1 <- Back_VaR_LSTM_1$Loss$Loss
Back_VaR_QL_LSTM_5 <- Back_VaR_LSTM_5$Loss$Loss

Back_VaR_FZ_LSTM_1 <- mean(FZLoss(df_lstm$Return, df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, 0.01))
Back_VaR_FZ_LSTM_5 <- mean(FZLoss(df_lstm$Return, df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, 0.05))

Back_VaR_NZ_LSTM_1 <- mean(NZ_deprecated(df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, df_lstm$Return, 0.01))
Back_VaR_NZ_LSTM_5 <- mean(NZ_deprecated(df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, df_lstm$Return, 0.05))

Back_VaR_AL_LSTM_1 <- mean(AL_deprecated(df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, df_lstm$Return, 0.01))
Back_VaR_AL_LSTM_5 <- mean(AL_deprecated(df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, df_lstm$Return, 0.05))

df_scores_lstm <- data.frame(
  Nível = c("1%", "5%"),
  QL = c(Back_VaR_QL_LSTM_1, Back_VaR_QL_LSTM_5),
  FZ = c(Back_VaR_FZ_LSTM_1, Back_VaR_FZ_LSTM_5),
  NZ = c(Back_VaR_NZ_LSTM_1, Back_VaR_NZ_LSTM_5),
  AL = c(Back_VaR_AL_LSTM_1, Back_VaR_AL_LSTM_5)
)

print(df_scores_lstm)


plot(df_lstm$Vol2_LSTM, type = 'l')
plot(df_lstm$VaR_LSTM_5, type = 'l')
plot(df_all_raw$VaR_GARCH_5, type = 'l')
