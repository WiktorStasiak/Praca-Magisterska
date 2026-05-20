"""
backtest/engine.py – Silnik backtestingu portfeli
==================================================
Moduł symuluje wyniki portfeli na danych out-of-sample (okres testowy
2024–2025) i oblicza standardowe metryki oceny strategii inwestycyjnych.

Metodologia backtestingu:
  • Wagi portfela ustalone raz (na początku okresu testowego) – "buy & hold"
  • Brak kosztów transakcyjnych i podatków (uproszczenie modelowe)
  • Stopa zwrotu portfela każdego dnia: r_p(t) = Σ w_i * r_i(t)
  • Skumulowana stopa zwrotu: C(T) = exp(Σ r_p(t)) - 1

Obliczane metryki:
  • Całkowity zwrot            : C(T) – ile zarobiliśmy łącznie
  • CAGR (roczna stopa wzrostu): (1 + C(T))^(252/T) - 1
  • Volatility (roczna)        : std(r_p) * √252
  • Wskaźnik Sharpe'a          : (CAGR - r_f) / vol
  • Max Drawdown               : maksymalne obsunięcie od szczytu
  • Calmar Ratio               : CAGR / |Max Drawdown|
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class BacktestEngine:
    """Wykonuje backtest portfela na zbiorze testowym i oblicza metryki.

    Klasa przyjmuje stałe wagi portfela (strategia kup-i-trzymaj bez
    rebalansowania) i symuluje dzienne stopy zwrotu na danych testowych.

    Attributes:
        risk_free: Dzienna stopa wolna od ryzyka (do obliczania Sharpe'a).
        trading_days: Liczba sesji giełdowych w roku.
    """

    def __init__(
        self,
        risk_free:    float = 0.045 / 252,
        trading_days: int   = 252,
    ) -> None:
        """Inicjalizuje silnik backtestingu.

        Args:
            risk_free: Dzienna stopa wolna od ryzyka.
            trading_days: Liczba sesji giełdowych w roku (252 dla USA).
        """
        self.risk_free:    float = risk_free
        self.trading_days: int   = trading_days

    # ── Symulacja portfela ─────────────────────────────────────────────

    def run(
        self,
        weights:      np.ndarray,
        test_returns: pd.DataFrame,
        name:         str = "Portfolio",
    ) -> pd.Series:
        """Symuluje portfel na zbiorze testowym i zwraca skumulowane zwroty.

        Dzienny zwrot portfela:
            r_p(t) = w' * r(t) = Σ_{i=1}^{N} w_i * r_i(t)

        Skumulowany zwrot po T dniach:
            C(T) = exp( Σ_{t=1}^{T} r_p(t) ) - 1

        Użycie logarytmicznych stóp zwrotu pozwala na ich prostą summację
        (addytywność), co jest niemożliwe dla stóp prostych.

        Args:
            weights: Wektor wag portfela, shape (n_assets,).
                     Musi sumować się do 1.
            test_returns: DataFrame log-stóp zwrotu z okresu testowego.
                          Shape: (n_days, n_assets).
            name: Etykieta portfela (używana w wykresach).

        Returns:
            Series skumulowanych stóp zwrotu portfela.
            Index: daty z test_returns. Wartości: ułamki (np. 0.15 = +15%).
        """
        # Dzienny zwrot portfela: iloczyn wektora wag i macierzy zwrotów
        # test_returns.values: (n_days, n_assets) @ weights: (n_assets,)
        # Wynik: (n_days,) – dzienny zwrot portfela dla każdej sesji
        daily_port_returns: np.ndarray = test_returns.values @ weights

        # Skumulowana stopa zwrotu: sumujemy log-zwroty, potem exp() - 1
        # cumsum() daje narastającą sumę → exp() daje mnożnik wartości portfela
        cumulative: pd.Series = pd.Series(
            np.exp(np.cumsum(daily_port_returns)) - 1,
            index=test_returns.index,
            name=name,
        )

        return cumulative

    def run_benchmark(
        self,
        benchmark_prices: pd.Series,
        test_start:       str,
        name:             str = "Benchmark",
    ) -> Optional[pd.Series]:
        """Oblicza skumulowane zwroty benchmarku (np. ^DJI) w okresie testowym.

        Benchmark służy jako punkt odniesienia: czy aktywna strategia
        portfelowa pobija pasywne trzymanie całego indeksu?

        Args:
            benchmark_prices: Series dziennych cen zamknięcia benchmarku.
            test_start: Data rozpoczęcia okresu testowego ('YYYY-MM-DD').
            name: Etykieta benchmarku w wykresach.

        Returns:
            Series skumulowanych zwrotów benchmarku, lub None jeśli
            brakuje wystarczająco dużo danych benchmarku.
        """
        # Wytnij ceny benchmarku do okresu testowego
        bench_test: pd.Series = benchmark_prices.loc[test_start:]

        if len(bench_test) < 10:
            return None

        # Log-stopy zwrotu benchmarku: r_m(t) = ln(P_m(t) / P_m(t-1))
        bench_returns: pd.Series = np.log(
            bench_test / bench_test.shift(1)
        ).dropna()

        cumulative: pd.Series = pd.Series(
            np.exp(np.cumsum(bench_returns.values)) - 1,
            index=bench_returns.index,
            name=name,
        )

        return cumulative

    def equal_weight(
        self,
        test_returns: pd.DataFrame,
        name:         str = "Równoważony (1/N)",
    ) -> pd.Series:
        """Symuluje portfel równoważony (1/N) jako alternatywny benchmark.

        Strategia 1/N przypisuje każdej spółce identyczną wagę 1/N.
        Badania (DeMiguel et al., 2009) pokazują, że 1/N jest zaskakująco
        trudny do pobicia przez optymalizowane portfele out-of-sample.

        Args:
            test_returns: DataFrame log-stóp zwrotu z okresu testowego.
            name: Etykieta portfela w wykresach.

        Returns:
            Series skumulowanych zwrotów portfela równoważonego.
        """
        n: int = test_returns.shape[1]
        # Równe wagi: każda spółka stanowi 1/N portfela
        equal_weights: np.ndarray = np.ones(n) / n
        return self.run(equal_weights, test_returns, name=name)

    # ── Metryki wydajności ─────────────────────────────────────────────

    def compute_metrics(
        self,
        cumulative:   pd.Series,
        test_returns: Optional[pd.DataFrame] = None,
        weights:      Optional[np.ndarray]   = None,
    ) -> dict[str, float]:
        """Oblicza kompletny zestaw metryk wydajności portfela.

        Definicje metryk:
        ─────────────────
        Całkowity zwrot:
            C = cum_returns.iloc[-1]

        CAGR (Compound Annual Growth Rate – Złożona Roczna Stopa Wzrostu):
            CAGR = (1 + C)^(252 / T) - 1
            gdzie T = liczba sesji w okresie testowym
            Interpretacja: jaka stała roczna stopa dałaby ten sam wynik końcowy

        Volatility (roczna zmienność):
            vol = std(r_p_daily) * √252
            gdzie r_p_daily = dzienne stopy zwrotu portfela
            Jest to roczne odchylenie standardowe – miara ryzyka

        Wskaźnik Sharpe'a:
            SR = (CAGR - r_f_annual) / vol
            Interpretacja: nadwyżkowy zwrot ponad stopę wolną na jednostkę ryzyka
            SR > 1.0  → dobry  |  SR > 2.0  → bardzo dobry

        Max Drawdown (MDD):
            MDD = min_t [ (1 + C(t)) / max_{s≤t}(1 + C(s)) - 1 ]
            Interpretacja: największe procentowe obsunięcie kapitału od szczytu
            MDD = -20% oznacza, że w najgorszym momencie straciliśmy 20% od szczytu

        Calmar Ratio:
            Calmar = CAGR / |MDD|
            Interpretacja: ile jednostek zwrotu na jednostkę maksymalnego obsunięcia
            Wyższy = lepsza relacja zysk/obsunięcie

        Args:
            cumulative: Series skumulowanych zwrotów (wyjście z run()).
            test_returns: Opcjonalnie: DataFrame dziennych zwrotów (do vol/Sharpe).
                          Jeśli None, oblicza dzienny zwrot z cumulative.
            weights: Opcjonalnie: wagi portfela (do obliczenia dziennych zwrotów).

        Returns:
            Słownik z metrykami: 'total_return', 'cagr', 'volatility',
            'sharpe', 'max_drawdown', 'calmar'.
        """
        T: int = len(cumulative)  # liczba sesji testowych

        # ── Całkowity zwrot ────────────────────────────────────────────
        total_return: float = float(cumulative.iloc[-1])

        # ── CAGR: annualizowany zwrot ──────────────────────────────────
        # Wzór: (wartość_końcowa / wartość_początkowa)^(252/T) - 1
        # 1 + total_return = mnożnik wartości portfela
        cagr: float = float((1.0 + total_return) ** (self.trading_days / T) - 1.0)

        # ── Dzienne zwroty portfela (do volatility i Sharpe) ──────────
        if test_returns is not None and weights is not None:
            # Preferujemy oryginalne dzienne zwroty (dokładniejsze)
            daily_rets: np.ndarray = test_returns.values @ weights
        else:
            # Alternatywnie: wyprowadzamy z kumulatywnych (mniej dokładne)
            port_value: pd.Series = 1.0 + cumulative  # wartość portfela
            daily_rets = port_value.pct_change().dropna().values

        # ── Volatility: roczna zmienność ───────────────────────────────
        # std(r_daily) * √252 → roczne odchylenie standardowe
        volatility: float = float(daily_rets.std() * np.sqrt(self.trading_days))

        # ── Wskaźnik Sharpe'a ─────────────────────────────────────────
        # (CAGR - r_f_roczna) / vol; +1e-10 chroni przed dzieleniem przez zero
        rf_annual: float = self.risk_free * self.trading_days
        sharpe: float    = float((cagr - rf_annual) / (volatility + 1e-10))

        # ── Max Drawdown ───────────────────────────────────────────────
        # Przekształcamy na wartość portfela (indeksowaną od 1.0)
        wealth: pd.Series  = 1.0 + cumulative
        # Running maximum: cummax() śledzi dotychczasowy szczyt
        running_max: pd.Series = wealth.cummax()
        # Drawdown w każdym dniu: jak daleko jesteśmy od szczytu
        drawdown: pd.Series = wealth / running_max - 1.0
        # Maksymalne obsunięcie (najgłębszy dołek)
        max_drawdown: float = float(drawdown.min())

        # ── Calmar Ratio ───────────────────────────────────────────────
        # Wyższy Calmar = lepsza relacja zysk/ryzyko obsunięcia
        calmar: float = float(cagr / (abs(max_drawdown) + 1e-10))

        return {
            "total_return": total_return,
            "cagr":         cagr,
            "volatility":   volatility,
            "sharpe":       sharpe,
            "max_drawdown": max_drawdown,
            "calmar":       calmar,
        }

    def print_comparison_table(
        self,
        results: dict[str, dict[str, float]],
    ) -> None:
        """Drukuje zestawienie metryk wszystkich strategii w tabeli.

        Args:
            results: Słownik {nazwa_strategii: słownik_metryk}.
                     Metryki: wyjście z compute_metrics().
        """
        strategies = list(results.keys())
        metrics_order = [
            ("total_return", "Całkowity zwrot",    "{:>9.2%}"),
            ("cagr",         "CAGR (roczny)",      "{:>9.2%}"),
            ("volatility",   "Volatility (rocz.)", "{:>9.2%}"),
            ("sharpe",       "Sharpe Ratio",        "{:>9.4f}"),
            ("max_drawdown", "Max Drawdown",        "{:>9.2%}"),
            ("calmar",       "Calmar Ratio",        "{:>9.4f}"),
        ]

        col_w = 14  # szerokość kolumny strategii

        print("\n" + "═" * 68)
        print("  WYNIKI BACKTESTINGU (out-of-sample, okres testowy)")
        print("═" * 68)

        # Nagłówek tabeli
        header = f"  {'Metryka':<24}"
        for s in strategies:
            # Skrócenie długich nazw do col_w znaków
            header += f"  {s[:col_w]:>{col_w}}"
        print(header)
        print("  " + "─" * (24 + (col_w + 2) * len(strategies)))

        # Wiersze metryk
        for key, label, fmt in metrics_order:
            row = f"  {label:<24}"
            for s in strategies:
                val = results[s].get(key, float("nan"))
                row += "  " + fmt.format(val).rjust(col_w)
            print(row)

        print("═" * 68)