"""
main.py – Główny skrypt uruchamiający optymalizator portfela DJIA
=================================================================
Uruchomienie:
    python main.py

Pełny potok obliczeniowy (pipeline):
    1. Pobierz dane cen zamknięcia 30 spółek DJIA z Yahoo Finance
    2. Oblicz log-stopy zwrotu; podziel na trening/walidację/test
    3. Zbuduj sekwencje wejściowe (lookback okna) dla modelu LSTM
    4. Wytrenuj LSTM z Monte Carlo Dropout
    5. Wykonaj próbkowanie MC → μ_LSTM i Σ_hybrid
    6. Oszacuj μ_CAPM z modelu Capital Asset Pricing Model
    7. Zoptymalizuj 3 portfele Markowitza (LSTM, CAPM, historyczny)
    8. Backtesting na danych 2024–2025 vs. benchmark DJIA
    9. Wygeneruj 7 wykresów i wydrukuj tabelę wyników

Zależności:
    pip install numpy pandas yfinance scikit-learn scipy
                tensorflow matplotlib

Struktura projektu:
    dji_optimizer/
    ├── config.py                  # stałe i parametry
    ├── main.py                    #
    ├── data/
    │   └── loader.py              # DataLoader, SequenceBuilder
    ├── models/
    │   └── lstm_model.py          # LSTMForecaster (MC Dropout)
    ├── optimization/
    │   └── portfolio.py           # MarkowitzOptimizer, CAPMEstimator
    ├── backtest/
    │   └── engine.py              # BacktestEngine
    └── visualization/
        └── plots.py               # Plotter (7 wykresów)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

# ── Importy z własnych modułów projektu ───────────────────────────────
from config import (
    TICKERS, BENCHMARK_TICKER,
    START_DATE, END_DATE,
    TRAIN_START, TRAIN_END,
    VALIDATION_START, VALIDATION_END,
    TEST_START, TEST_END,
    LOOKBACK, LSTM_UNITS, DROPOUT_RATE, EPOCHS, BATCH_SIZE,
    MC_SAMPLES, RISK_FREE, TRADING_DAYS, N_RESTARTS, SEED,
)
from loader import DataLoader, SequenceBuilder
from lstm_model import LSTMForecaster
from portfolio import MarkowitzOptimizer, CAPMEstimator
from engine import BacktestEngine
from plots import Plotter


# ── Ustawienie ziaren losowości dla pełnej reprodukowalności ───────────
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main() -> None:
    """Uruchamia pełny potok: dane → model → optymalizacja → backtesting → wykresy."""

    print("\n" + "═" * 68)
    print("  DJIA PORTFOLIO OPTIMIZER")
    print("  LSTM + Monte Carlo Dropout  vs.  CAPM  vs.  Klasyczny Markowitz")
    print("  Dow Jones Industrial Average – 30 spółek, lata 2011–2025")
    print("═" * 68)

    plotter = Plotter(output_dir=".")   # wykresy zapisywane w bieżącym katalogu

    # ══════════════════════════════════════════════════════════════════
    # KROK 1: POBIERANIE I PRZYGOTOWANIE DANYCH
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 68)
    print("KROK 1 ▶ Pobieranie danych z Yahoo Finance")
    print("─" * 68)

    loader = DataLoader(TICKERS, START_DATE, END_DATE)

    # Ceny zamknięcia (korygowane o splity i dywidendy)
    prices: pd.DataFrame = loader.download_prices()

    # Benchmark: indeks ^DJI (Dow Jones Industrial Average)
    benchmark_prices: pd.Series = loader.download_benchmark(BENCHMARK_TICKER)

    # Log-stopy zwrotu: r_t = ln(P_t / P_{t-1})
    returns: pd.DataFrame = loader.compute_returns(prices)

    # Wyrównaj indeks cen do indeksu stóp (usuwa pierwszy wiersz)
    prices = prices.loc[returns.index]

    asset_names: list[str] = list(prices.columns)
    n_assets:    int        = len(asset_names)
    print(f"\n  ✓ Aktywów w portfelu: {n_assets}")

    # ── Podział na 3 rozłączne okresy ─────────────────────────────────
    print("\n  Podział danych:")
    split: dict[str, pd.DataFrame] = loader.split_data(
        prices, returns,
        TRAIN_START, TRAIN_END,
        VALIDATION_START, VALIDATION_END,
        TEST_START, TEST_END,
    )
    train_prices:  pd.DataFrame = split["train_prices"]
    train_returns: pd.DataFrame = split["train_returns"]
    val_prices:    pd.DataFrame = split["val_prices"]
    val_returns:   pd.DataFrame = split["val_returns"]
    test_prices:   pd.DataFrame = split["test_prices"]
    test_returns:  pd.DataFrame = split["test_returns"]

    # Wykres korelacji historycznej (z danych treningowych)
    print("\n  Generowanie macierzy korelacji...")
    plotter.plot_correlation_matrix(
        train_returns,
        title="Historyczna macierz korelacji DJIA – dane treningowe (2011–2021)"
    )

    # ══════════════════════════════════════════════════════════════════
    # KROK 2: PRZYGOTOWANIE SEKWENCJI DLA LSTM
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 68)
    print("KROK 2 ▶ Budowanie sekwencji wejściowych dla LSTM")
    print("─" * 68)

    builder = SequenceBuilder(lookback=LOOKBACK)

    # Sekwencje treningowe – scaler dopasowywany TYLKO na train_prices
    X_train, y_train = builder.build_sequences(train_prices, fit=True)

    # Sekwencje walidacyjne – używamy scalera z treningu (bez refitu!)
    # Łączymy ostatnie LOOKBACK dni treningu z danymi walidacyjnymi,
    # aby zapewnić ciągłość sekwencji na granicy podziału
    val_context: pd.DataFrame = pd.concat([train_prices.iloc[-LOOKBACK:], val_prices])
    X_val, y_val = builder.build_sequences(val_context, fit=False)

    print(f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}    |  y_val:   {y_val.shape}")
    print(f"  Lookback = {LOOKBACK} sesji ({LOOKBACK//21} mies.);  "
          f"features = {n_assets} spółek DJIA")

    # ══════════════════════════════════════════════════════════════════
    # KROK 3: BUDOWA I TRENING MODELU LSTM
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 68)
    print("KROK 3 ▶ Budowanie i trening modelu LSTM + MC Dropout")
    print("─" * 68)

    forecaster = LSTMForecaster(
        n_features=n_assets,
        lookback=LOOKBACK,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE,
    )
    forecaster.build()
    forecaster.model.summary()

    history = forecaster.train(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # Wykres 1: krzywa uczenia
    plotter.plot_training_history(history)

    # ══════════════════════════════════════════════════════════════════
    # KROK 4: MONTE CARLO DROPOUT – PROGNOZOWANIE I ESTYMACJA RYZYKA
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 68)
    print(f"KROK 4 ▶ Monte Carlo Dropout ({MC_SAMPLES} próbek)")
    print("─" * 68)

    # Wejście MC: ostatnie LOOKBACK dni danych treningowych
    # (okno tuż przed początkiem okresu walidacyjnego/testowego)
    last_train_window: np.ndarray = (
        builder.scaler.transform(train_prices.values)[-LOOKBACK:]
    )  # shape: (LOOKBACK, n_assets)

    # Dodaj wymiar batch: (1, LOOKBACK, n_assets)
    X_mc_input: np.ndarray = last_train_window[np.newaxis, :, :]

    # MC Dropout: S=MC_SAMPLES niezależnych przebiegów z aktywnym Dropout
    mc_returns: np.ndarray = forecaster.mc_dropout_predict(
        X_mc_input, MC_SAMPLES, builder.scaler
    )
    print(f"  mc_returns.shape = {mc_returns.shape}  "
          f"(próbki × aktywa)")

    # Wykres 2: rozkłady MC Dropout
    plotter.plot_mc_distributions(mc_returns, asset_names, max_assets=10)

    # ── Hybrydowa macierz kowariancji ─────────────────────────────────
    # μ_LSTM: średnia MC Dropout → oczekiwane stopy zwrotu wg modelu
    # Σ_hybrid: korelacje historyczne × wariancje MC (patrz lstm_model.py)
    mu_lstm, cov_hybrid = forecaster.compute_mc_statistics(
        mc_returns,
        hist_returns=train_returns,   # ← historyczne korelacje z 11 lat treningu
    )

    print(f"\n  μ_LSTM (dzienna, top 5):")
    top5_idx = np.argsort(mu_lstm)[::-1][:5]
    for i in top5_idx:
        print(f"    {asset_names[i]:6}: {mu_lstm[i]:+.6f}  "
              f"({mu_lstm[i] * 252:+.2%} rocznie)")

    # ══════════════════════════════════════════════════════════════════
    # KROK 5: ESTYMACJA CAPM
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 68)
    print("KROK 5 ▶ Estymacja oczekiwanych zwrotów metodą CAPM")
    print("─" * 68)

    capm = CAPMEstimator(risk_free=RISK_FREE)

    # Stopy zwrotu benchmarku w okresie treningowym
    bench_returns_full: pd.Series = CAPMEstimator.compute_benchmark_returns(
        benchmark_prices
    )
    bench_train: pd.Series = bench_returns_full.loc[TRAIN_START:TRAIN_END]

    # μ_CAPM = r_f + β_i * (μ_m - r_f)
    # Σ_hist = historyczna macierz kowariancji z danych treningowych
    mu_capm, cov_hist = capm.estimate(train_returns, bench_train)

    # Klasyczny Markowitz używa też historycznych statystyk
    mu_hist: np.ndarray = train_returns.mean().values   # prosta średnia historyczna

    # ══════════════════════════════════════════════════════════════════
    # KROK 6: OPTYMALIZACJA PORTFELI MARKOWITZA
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 68)
    print("KROK 6 ▶ Optymalizacja portfeli – maksymalizacja Sharpe'a")
    print("─" * 68)

    optimizer = MarkowitzOptimizer(
        risk_free=RISK_FREE,
        trading_days=TRADING_DAYS,
        n_restarts=N_RESTARTS,
    )

    # ── Portfel 1: Hybrydowy (LSTM + MC Dropout) ──────────────────────
    # μ z MC Dropout; Σ = korelacje historyczne × wariancje MC
    print("\n  [1/3] Portfel Hybrydowy (LSTM + MC Dropout)...")
    weights_hybrid: np.ndarray = optimizer.optimize(mu_lstm, cov_hybrid)
    ret_h, vol_h, sr_h = optimizer.portfolio_performance(
        weights_hybrid, mu_lstm, cov_hybrid
    )
    print(f"  Sharpe = {sr_h:.4f} | Zwrot = {ret_h:.2%} | Vol = {vol_h:.2%}")

    # ── Portfel 2: CAPM + Markowitz ───────────────────────────────────
    # μ z CAPM (r_f + β*(μ_m - r_f)); Σ = historyczna
    print("\n  [2/3] Portfel CAPM + Markowitz...")
    weights_capm: np.ndarray = optimizer.optimize(mu_capm, cov_hist)
    ret_c, vol_c, sr_c = optimizer.portfolio_performance(
        weights_capm, mu_capm, cov_hist
    )
    print(f"  Sharpe = {sr_c:.4f} | Zwrot = {ret_c:.2%} | Vol = {vol_c:.2%}")

    # ── Portfel 3: Klasyczny Markowitz (historyczny) ───────────────────
    # μ = prosta średnia historyczna; Σ = historyczna macierz kowariancji
    print("\n  [3/3] Portfel Klasyczny (historyczny Markowitz)...")
    weights_classic: np.ndarray = optimizer.optimize(mu_hist, cov_hist)
    ret_k, vol_k, sr_k = optimizer.portfolio_performance(
        weights_classic, mu_hist, cov_hist
    )
    print(f"  Sharpe = {sr_k:.4f} | Zwrot = {ret_k:.2%} | Vol = {vol_k:.2%}")

    # Wykres 3: porównanie wag
    plotter.plot_portfolio_weights(
        weights_dict={
            "Hybrydowy (LSTM+MC)":    weights_hybrid,
            "CAPM + Markowitz":        weights_capm,
            "Klasyczny (Hist. MVO)":   weights_classic,
        },
        asset_names=asset_names,
    )

    # ══════════════════════════════════════════════════════════════════
    # KROK 7: BACKTESTING NA DANYCH TESTOWYCH (2024–2025)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 68)
    print("KROK 7 ▶ Backtesting (out-of-sample, 2024–2025)")
    print("─" * 68)

    engine = BacktestEngine(risk_free=RISK_FREE, trading_days=TRADING_DAYS)

    # Skumulowane zwroty każdej strategii
    cum_hybrid:  pd.Series = engine.run(
        weights_hybrid,  test_returns, name="Hybrydowy (LSTM+MC)"
    )
    cum_capm:    pd.Series = engine.run(
        weights_capm,    test_returns, name="CAPM + Markowitz"
    )
    cum_classic: pd.Series = engine.run(
        weights_classic, test_returns, name="Klasyczny (Hist. MVO)"
    )
    cum_equal:   pd.Series = engine.equal_weight(test_returns, name="Równoważony (1/N)")

    # Benchmark DJIA z Yahoo Finance lub portfel równoważony jako fallback
    cum_bench_raw: pd.Series | None = engine.run_benchmark(
        benchmark_prices, TEST_START, name="DJIA (benchmark)"
    )

    if cum_bench_raw is not None and len(cum_bench_raw) > 10:
        # Wyrównaj daty benchmarku do dat testowego zwrotu
        cum_bench: pd.Series = cum_bench_raw.reindex(test_returns.index).ffill().bfill()
        cum_bench.name = "DJIA (benchmark)"
        print("  ✓ Benchmark: DJIA (^DJI z Yahoo Finance)")
    else:
        # Fallback: indeks DJIA niedostępny – używamy 1/N jako odniesienia
        cum_bench = cum_equal.copy()
        cum_bench.name = "Równoważony 1/N (fallback)"
        print("  [INFO] Benchmark DJIA niedostępny – używam portfela 1/N")

    # ── Słowniki do wykresów i metryk ─────────────────────────────────
    # Klucze MUSZĄ odpowiadać kolorom w visualization/plots.py (COLORS dict)
    portfolios_cum: dict[str, pd.Series] = {
        "hybrid":    cum_hybrid,
        "capm":      cum_capm,
        "classic":   cum_classic,
        "benchmark": cum_bench,
        "equal":     cum_equal,
    }

    # Dzienne zwroty portfeli (do rolling Sharpe)
    def _daily(weights: np.ndarray, name: str) -> pd.Series:
        """Oblicza dzienne log-zwroty portfela."""
        s = pd.Series(
            test_returns.values @ weights,
            index=test_returns.index,
            name=name,
        )
        return s

    bench_daily: pd.Series = (
        np.log(benchmark_prices / benchmark_prices.shift(1))
        .dropna()
        .reindex(test_returns.index)
        .fillna(0.0)
        .rename("DJIA (benchmark)")
    )

    portfolios_daily: dict[str, pd.Series] = {
        "hybrid":    _daily(weights_hybrid,  "Hybrydowy (LSTM+MC)"),
        "capm":      _daily(weights_capm,    "CAPM + Markowitz"),
        "classic":   _daily(weights_classic, "Klasyczny (Hist. MVO)"),
        "benchmark": bench_daily,
        "equal":     _daily(np.ones(n_assets) / n_assets, "Równoważony (1/N)"),
    }

    # ── Oblicz metryki dla każdej strategii ───────────────────────────
    metrics_hybrid:  dict = engine.compute_metrics(
        cum_hybrid,  test_returns, weights_hybrid
    )
    metrics_capm:    dict = engine.compute_metrics(
        cum_capm,    test_returns, weights_capm
    )
    metrics_classic: dict = engine.compute_metrics(
        cum_classic, test_returns, weights_classic
    )
    metrics_bench:   dict = engine.compute_metrics(cum_bench)
    metrics_equal:   dict = engine.compute_metrics(
        cum_equal,   test_returns, np.ones(n_assets) / n_assets
    )

    # ── Tabela porównawcza ─────────────────────────────────────────────
    engine.print_comparison_table({
        "LSTM+MC":    metrics_hybrid,
        "CAPM":       metrics_capm,
        "Hist. MVO":  metrics_classic,
        "DJIA":       metrics_bench,
        "1/N":        metrics_equal,
    })

    # ══════════════════════════════════════════════════════════════════
    # KROK 8: WYKRESY WYNIKÓW
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 68)
    print("KROK 8 ▶ Generowanie wykresów wyników")
    print("─" * 68)

    # Wykres 5: skumulowane zwroty (główny wykres backtestingu)
    plotter.plot_cumulative_returns(portfolios_cum)

    # Wykres 6: rolling Sharpe (stabilność strategii w czasie)
    plotter.plot_rolling_sharpe(portfolios_daily, window=63, risk_free_daily=RISK_FREE)

    # Wykres 7: drawdown (obsunięcia kapitału)
    plotter.plot_drawdown(portfolios_cum)

    # ── Podsumowanie wag ──────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("PODSUMOWANIE OPTYMALNYCH WAG PORTFELI")
    print("─" * 68)
    print(f"\n  {'Spółka':8}  {'LSTM+MC':>10}  {'CAPM':>10}  {'Hist.MVO':>10}")
    print(f"  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*10}")

    for i, name in enumerate(asset_names):
        h  = weights_hybrid[i]
        c  = weights_capm[i]
        cl = weights_classic[i]
        # Wyświetl tylko spółki z wagą > 1% w przynajmniej jednej strategii
        if max(h, c, cl) > 0.01:
            print(f"  {name:8}  {h:>10.2%}  {c:>10.2%}  {cl:>10.2%}")

    print(f"\n  {'SUMA':8}  {weights_hybrid.sum():>10.2%}  "
          f"{weights_capm.sum():>10.2%}  {weights_classic.sum():>10.2%}")

    print("\n" + "═" * 68)
    print("  ✓ Pipeline zakończony pomyślnie.")
    print("  ✓ Wykresy zapisane: 01–07 *.png")
    print("═" * 68 + "\n")


# ── Entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()