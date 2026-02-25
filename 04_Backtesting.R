library(dplyr)
library(readr)
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

table(df_lstm$Exceed_VaR_LSTM_1)
table(df_lstm$Exceed_VaR_LSTM_5)

mean(df_lstm$Return < df_lstm$VaR_LSTM_1)  # deveria ~0.01
mean(df_lstm$Return < df_lstm$VaR_LSTM_5)  # deveria ~0.05

Back_VaR_LSTM_1 <- BacktestVaR(df_lstm$Return, df_lstm$VaR_LSTM_1, 0.01)
Back_VaR_LSTM_5 <- BacktestVaR(df_lstm$Return, df_lstm$VaR_LSTM_5, 0.05)

VaR_VQR(df_lstm$Return, df_lstm$VaR_LSTM_1, 0.01)
VaR_VQR(df_lstm$Return, df_lstm$VaR_LSTM_5, 0.05)

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

# ============================
# ES Backtests (CoC, ER, ESR)
# ============================
vol_lstm <- df_lstm$Vol_LSTM

Back_ES_CoC_LSTM_1 <- cc_backtest(df_lstm$Return, df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, vol_lstm, 0.01)
Back_ES_CoC_LSTM_5 <- cc_backtest(df_lstm$Return, df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, vol_lstm, 0.05)

Back_ES_ER_LSTM_1 <- er_backtest(df_lstm$Return, df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, vol_lstm)
Back_ES_ER_LSTM_5 <- er_backtest(df_lstm$Return, df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, vol_lstm)

Back_ES_ESR_LSTM_1_V1 <- esr_backtest(df_lstm$Return, df_lstm$VaR_LSTM_1, df_lstm$ES_LSTM_1, alpha = 0.01, version = 1, B = 0)
Back_ES_ESR_LSTM_5_V1 <- esr_backtest(df_lstm$Return, df_lstm$VaR_LSTM_5, df_lstm$ES_LSTM_5, alpha = 0.05, version = 1, B = 0)

# ============================
# TABELA FINAL (p-valores)
# ============================

pVQR_1 <- vqr1
pVQR_5 <- vqr5

df_pvals_tests <- data.frame(
  Nivel = c("1%", "5%"),

  UC  = c(Back_VaR_LSTM_1$LRuc["Pvalue"], Back_VaR_LSTM_5$LRuc["Pvalue"]),
  CC  = c(Back_VaR_LSTM_1$LRcc["Pvalue"], Back_VaR_LSTM_5$LRcc["Pvalue"]),
  DQ  = c(Back_VaR_LSTM_1$DQ$pvalue,      Back_VaR_LSTM_5$DQ$pvalue),
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


###checks

cor(abs(df_lstm$Return), sqrt(df_lstm$Vol2_LSTM))  # você tem Vol_LSTM no df_lstm


sig0  <- sqrt(df_lstm$Vol2_LSTM)
sigm1 <- dplyr::lag(sig0, 1)
sigp1 <- dplyr::lead(sig0, 1)

c(
  corr_t   = cor(abs(df_lstm$Return), sig0,  use="complete.obs"),
  corr_tm1 = cor(abs(df_lstm$Return), sigm1, use="complete.obs"),
  corr_tp1 = cor(abs(df_lstm$Return), sigp1, use="complete.obs")
)

summary(df_lstm$Return)
summary(sqrt(df_lstm$Vol2_LSTM))


cor(abs(df_lstm$Return), sqrt(df_lstm$RV_true), use="complete.obs")


sig <- sqrt(pmax(df_lstm$Vol2_LSTM, 1e-12))
sig_floor <- quantile(sig, 0.01, na.rm = TRUE)

df_lstm2 <- df_lstm %>%
  mutate(
    Vol2_LSTM_adj = pmax(Vol2_LSTM, sig_floor^2),
    Vol_LSTM_adj  = sqrt(Vol2_LSTM_adj)
  )
summary(df_lstm2$Vol_LSTM_adj)



# 1) Escala do retorno
summary(df_lstm$Return)
sd_ret <- sd(df_lstm$Return, na.rm = TRUE)

# 2) Escala da vol prevista (sqrt da variância)
sig_hat <- sqrt(pmax(df_lstm$Vol2_LSTM, 1e-12))
summary(sig_hat)
sd_sig <- sd(sig_hat, na.rm = TRUE)

cat("\nSD(Return) =", sd_ret, "\n")
cat("Median(|Return|) =", median(abs(df_lstm$Return), na.rm=TRUE), "\n")
cat("Median(sigma_hat) =", median(sig_hat, na.rm=TRUE), "\n")
cat("Mean(sigma_hat) =", mean(sig_hat, na.rm=TRUE), "\n")




m_ret2 <- mean(df_lstm$Return^2, na.rm=TRUE)
m_rvhat <- mean(df_lstm$Vol2_LSTM, na.rm=TRUE)

cat("\nMean(Return^2) =", m_ret2, "\n")
cat("Mean(RV_hat)    =", m_rvhat, "\n")
cat("Ratio mean(Return^2)/mean(RV_hat) =", m_ret2 / m_rvhat, "\n")



sig_hat <- sqrt(pmax(df_lstm$Vol2_LSTM, 1e-12))

corr_pct <- cor(abs(df_lstm$Return), sig_hat, use="complete.obs")
corr_dec <- cor(abs(df_lstm$Return/100), sig_hat, use="complete.obs")

c(corr_pct = corr_pct, corr_dec = corr_dec)



library(dplyr)
library(readr)
library(lubridate)

df_full <- read_csv("GARCH_LSTM_1.csv", show_col_types = FALSE) %>%
  mutate(Date = ymd(Date))

df_lstm <- read_csv("VaR_ES_LSTM_df_full.csv", show_col_types = FALSE) %>%
  mutate(Date = ymd(Date))

# checar nomes das colunas
print(names(df_lstm))
print(names(df_full))

df_lstm2 <- df_lstm %>%
  left_join(df_full %>% select(Date, RV_true), by = "Date") %>%
  mutate(
    sig_hat  = sqrt(pmax(RV_hat,  1e-12)),  # variância prevista do LSTM (do seu CSV)
    sig_true = sqrt(pmax(RV_true, 1e-12))   # variância realizada verdadeira
  )

c(
  corr_absret_sigtrue = cor(abs(df_lstm2$Return), df_lstm2$sig_true, use="complete.obs"),
  corr_absret_sighat  = cor(abs(df_lstm2$Return), df_lstm2$sig_hat,  use="complete.obs"),
  corr_rv             = cor(df_lstm2$RV_hat, df_lstm2$RV_true,        use="complete.obs")
)
