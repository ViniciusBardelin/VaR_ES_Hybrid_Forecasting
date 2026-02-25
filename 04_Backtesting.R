library(dplyr)
library(readr)
library(GAS)
library(lubridate)
library(esback)
source("Function_VaR_VQR.R")
source("Optimizations.R")

df_lstm <- read_csv("VaR_ES_LSTM_TESTE_2.csv", show_col_types = FALSE) %>%
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

mean(df_lstm$Return < df_lstm$VaR_LSTM_1)  
mean(df_lstm$Return < df_lstm$VaR_LSTM_5) 

# VaR Backtests
Back_VaR_LSTM_1 <- BacktestVaR(df_lstm$Return, df_lstm$VaR_LSTM_1, 0.01)
Back_VaR_LSTM_5 <- BacktestVaR(df_lstm$Return, df_lstm$VaR_LSTM_5, 0.05)

pVQR_1 <- VaR_VQR(df_lstm$Return, df_lstm$VaR_LSTM_1, 0.01)
pVQR_5 <- VaR_VQR(df_lstm$Return, df_lstm$VaR_LSTM_5, 0.05)

# ES Backtests (CoC, ER, ESR)
vol_lstm <- df_lstm$Vol_LSTM

Back_ES_CoC_LSTM_1 <- cc_backtest(df_lstm$Return, df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, vol_lstm, 0.01)
Back_ES_CoC_LSTM_5 <- cc_backtest(df_lstm$Return, df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, vol_lstm, 0.05)

Back_ES_ER_LSTM_1 <- er_backtest(df_lstm$Return, df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, vol_lstm)
Back_ES_ER_LSTM_5 <- er_backtest(df_lstm$Return, df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, vol_lstm)

Back_ES_ESR_LSTM_1_V1 <- esr_backtest(df_lstm$Return, df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, alpha = 0.01, version = 1, B = 0)
Back_ES_ESR_LSTM_5_V1 <- esr_backtest(df_lstm$Return, df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, alpha = 0.05, version = 1, B = 0)

# Table (p-values)
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
