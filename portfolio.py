"""
optimization/portfolio.py – Optymalizacja portfela: Markowitz + CAPM
======================================================================
Moduł implementuje dwa podejścia do szacowania oczekiwanych stóp zwrotu:

1. KLASYCZNY MARKOWITZ (Historical MVO)
   ─────────────────────────────────────
   Oczekiwane zwroty μ i macierz kowariancji Σ szacowane są bezpośrednio
   z historycznych stóp zwrotu (próbkowe momenty statystyczne).
   Ograniczenie: "przeszłość nie musi powtarzać przyszłości" –
   estymatory próbkowe są niestabilne dla krótkiego horyzontu.

2. CAPM (Capital Asset Pricing Model)
   ─────────────────────────────────────
   μ_i = r_f + β_i * (μ_m - r_f)

   gdzie:
     r_f     = stopa wolna od ryzyka (T-Bill)
     μ_m     = oczekiwana stopa zwrotu rynku (benchmarku)
     β_i     = Beta spółki i = Cov(r_i, r_m) / Var(r_m)
     (μ_m - r_f) = premia za ryzyko rynkowe (market risk premium)

   Beta (β) mierzy wrażliwość spółki na ruchy rynku:
     β = 1.0 → spółka zachowuje się jak rynek
     β > 1.0 → spółka amplifikuje ruchy rynku (bardziej ryzykowna)
     β < 1.0 → spółka jest mniej wrażliwa (defensywna)
     β < 0.0 → spółka rośnie gdy rynek spada (hedge)

3. PORTFEL HYBRYDOWY (LSTM + MC Dropout)
   ─────────────────────────────────────
   μ i Σ z rozkładu Monte Carlo Dropout (patrz: models/lstm_model.py)

Referencje:
  Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
  Sharpe, W. (1964). Capital Asset Prices. Journal of Finance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult


class MarkowitzOptimizer:
    """Optymalizator portfela metodą Mean-Variance (Markowitz, 1952).

    Cel optymalizacji: znalezienie wag portfela maksymalizujących
    wskaźnik Sharpe'a na granicy efektywnej.

    Wskaźnik Sharpe'a (roczny):
        SR = (μ_p - r_f) / σ_p

    gdzie:
        μ_p = roczna oczekiwana stopa zwrotu portfela = w' * μ * 252
        σ_p = roczne odchylenie standardowe = √(w' * Σ * w * 252)
        r_f = roczna stopa wolna od ryzyka
        w   = wektor wag portfela (w' * 1 = 1, w_i ≥ 0)

    Ograniczenia optymalizacji:
        • Σ w_i = 1  (portfel w pełni zainwestowany)
        • w_i ≥ 0    (zakaz krótkiej sprzedaży – "long-only")

    Problem jest nieliniowy, więc używamy SLSQP (Sequential Least Squares
    Programming) z wieloma losowymi punktami startowymi (multi-start),
    co zmniejsza ryzyko utknięcia w minimum lokalnym.

    Attributes:
        risk_free: Dzienna stopa wolna od ryzyka.
        trading_days: Liczba sesji giełdowych w roku (252 dla USA).
        n_restarts: Liczba losowych punktów startowych optymalizacji.
    """

    def __init__(
        self,
        risk_free:    float = 0.045 / 252,
        trading_days: int   = 252,
        n_restarts:   int   = 20,
    ) -> None:
        """Inicjalizuje parametry optymalizatora.

        Args:
            risk_free: Dzienna stopa wolna od ryzyka.
                       Roczna stopa / 252 = dzienna stopa.
                       Domyślnie: 4.5% rocznie / 252 dni.
            trading_days: Liczba sesji giełdowych w roku.
                          Konwencja USA: 252 (NYSE/NASDAQ).
            n_restarts: Liczba losowych punktów startowych.
                        Więcej restartów → lepsze optimum globalne,
                        ale dłuższy czas obliczeń.
        """
        self.risk_free:    float = risk_free
        self.trading_days: int   = trading_days
        self.n_restarts:   int   = n_restarts

    def portfolio_performance(
        self,
        weights: np.ndarray,
        mu:      np.ndarray,
        cov:     np.ndarray,
    ) -> tuple[float, float, float]:
        """Oblicza roczne parametry wydajności portfela.

        Wzory:
            Dzienny zwrot portfela:    μ_p = w' * μ  = Σ w_i * μ_i
            Dzienna wariancja:         σ²_p = w' * Σ * w
            Roczny zwrot:              R_p = μ_p * T
            Roczne odchylenie (vol):   V_p = σ_p * √T
            Wskaźnik Sharpe'a (roczny): SR = (R_p - r_f * T) / V_p

        gdzie T = trading_days (252).

        Args:
            weights: Wektor wag portfela, shape (n_assets,).
                     Musi sumować się do 1.
            mu: Dzienny wektor oczekiwanych zwrotów, shape (n_assets,).
            cov: Dzienna macierz kowariancji, shape (n_assets, n_assets).

        Returns:
            Krotka (ret_annual, vol_annual, sharpe):
                ret_annual: Roczna oczekiwana stopa zwrotu [jako ułamek].
                vol_annual: Roczne odchylenie standardowe (ryzyko) [ułamek].
                sharpe: Wskaźnik Sharpe'a (wyższy = lepszy).
        """
        T = self.trading_days

        # Dzienny oczekiwany zwrot portfela: iloczyn skalarny wag i zwrotów
        ret_daily: float = float(np.dot(weights, mu))

        # Dzienna wariancja portfela: forma kwadratowa w' * Σ * w
        # Uwzględnia nie tylko wariancje spółek, ale też ich korelacje
        var_daily: float = float(weights @ cov @ weights)

        # Roczna annualizacja (standardowa konwencja finansów):
        #   zwrot  × T (liniowa skala)
        #   vol    × √T (pierwiastkowa skala – prawo pierwiastka)
        ret_annual: float = ret_daily * T
        vol_annual: float = np.sqrt(var_daily) * np.sqrt(T)

        # Wskaźnik Sharpe'a: nadwyżkowy zwrot ponad r_f na jednostkę ryzyka
        # +1e-10: epsilon chroni przed dzieleniem przez zero (gdy vol → 0)
        sharpe: float = (
            (ret_annual - self.risk_free * T) / (vol_annual + 1e-10)
        )

        return ret_annual, vol_annual, sharpe

    def optimize(
        self,
        mu:  np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Optymalizuje wagi portfela maksymalizując wskaźnik Sharpe'a.

        Strategia multi-start: uruchamiamy SLSQP z n_restarts różnych
        losowych punktów startowych (rozkład Dirichlet zapewnia,
        że punkt startowy jest zawsze prawidłowym portfelem: Σw=1, w≥0).
        Zachowujemy wynik z najwyższym Sharpe'em.

        Args:
            mu: Dzienny wektor oczekiwanych zwrotów, shape (n_assets,).
            cov: Dzienna macierz kowariancji, shape (n_assets, n_assets).

        Returns:
            Optymalny wektor wag portfela, shape (n_assets,).
            Zawsze spełnia: Σw=1, 0≤w_i≤1.

        Raises:
            RuntimeError: Jeśli żaden restart optymalizacji nie zakończy się sukcesem.
        """
        n: int = len(mu)

        # Ograniczenie równościowe: suma wag musi być równa 1
        # (portfel w pełni zainwestowany – brak gotówki)
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]

        # Ograniczenia nierównościowe: każda waga w przedziale [0, 1]
        # (zakaz krótkiej sprzedaży = long-only portfolio)
        bounds = [(0.0, 1.0)] * n

        # Funkcja celu: minimalizujemy UJEMNY Sharpe (scipy minimalizuje)
        def neg_sharpe(w: np.ndarray) -> float:
            """Zwraca ujemny wskaźnik Sharpe'a (do minimalizacji)."""
            _, _, sr = self.portfolio_performance(w, mu, cov)
            return -sr

        best_result: Optional[OptimizeResult] = None

        for restart_idx in range(self.n_restarts):
            # Punkt startowy: losowy portfel z rozkładu Dirichleta
            # Dirichlet(α=1) = rozkład jednostajny na sympleksie
            w0: np.ndarray = np.random.dirichlet(np.ones(n))

            try:
                result: OptimizeResult = minimize(
                    neg_sharpe,
                    w0,
                    method="SLSQP",        # Sequential Least Squares Programming
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        "maxiter": 2000,   # maksymalna liczba iteracji
                        "ftol": 1e-10,     # tolerancja zbieżności funkcji celu
                    },
                )

                # Zachowaj wynik jeśli jest lepszy (mniejszy neg_sharpe)
                if best_result is None or result.fun < best_result.fun:
                    best_result = result

            except Exception as exc:
                # Ignorujemy nieudane starty – kontynuujemy z pozostałymi
                if restart_idx == self.n_restarts - 1 and best_result is None:
                    raise RuntimeError(
                        f"Wszystkie {self.n_restarts} restartów optymalizacji "
                        f"zakończyły się niepowodzeniem: {exc}"
                    )

        # Klipowanie numeryczne: usuwa ewentualne drobne ujemne wagi (artefakty)
        weights: np.ndarray = np.clip(best_result.x, 0.0, 1.0)

        # Renormalizacja: zapewniamy dokładną sumę = 1 po klipowaniu
        weights /= weights.sum()

        return weights


class CAPMEstimator:
    """Szacuje oczekiwane stopy zwrotu spółek metodą CAPM.

    Model CAPM (Sharpe, 1964) zakłada, że jedynym czynnikiem ryzyka
    systematycznego jest rynek (benchmark). Premia za ryzyko każdej
    spółki jest proporcjonalna do jej ekspozycji na rynek (beta).

    Wzór CAPM:
        E[r_i] = r_f + β_i * (E[r_m] - r_f)

    Interpretacja:
        r_f              = "bezpieczna" minimalna stopa zwrotu (T-Bill)
        E[r_m] - r_f     = premia za ryzyko rynkowe (historycznie ~5-7% w USA)
        β_i              = "dźwignia rynkowa" spółki i
        β_i * (E[r_m]-r_f) = premia za ryzyko specyficzne dla spółki i

    Ograniczenia CAPM:
        • Zakłada jeden czynnik ryzyka (tylko rynek)
        • Beta historyczna ≠ beta przyszła
        • Nie uwzględnia czynników size, value, momentum (vide Fama-French)

    Attributes:
        risk_free: Dzienna stopa wolna od ryzyka.
    """

    def __init__(self, risk_free: float = 0.045 / 252) -> None:
        """Inicjalizuje estymator CAPM.

        Args:
            risk_free: Dzienna stopa wolna od ryzyka (r_f / 252).
        """
        self.risk_free: float = risk_free

    def estimate(
        self,
        train_returns:    pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Szacuje μ_CAPM i Σ_hist na podstawie danych treningowych.

        Krok 1 – Oblicz Beta każdej spółki:
            β_i = Cov(r_i, r_m) / Var(r_m)

        Krok 2 – Oblicz E[r_i] ze wzoru CAPM:
            E[r_i] = r_f + β_i * (μ_m - r_f)

        Krok 3 – Macierz kowariancji Σ pochodzi z danych historycznych
            (estymacja próbkowa – ten sam Σ co w klasycznym Markowitza).

        Wyrównanie indeksów (align): upewniamy się, że stopy zwrotu
        spółek i benchmarku pokrywają dokładnie te same daty sesji.

        Args:
            train_returns: DataFrame log-stóp zwrotu spółek (trening).
                           Kolumny: nazwy spółek. Index: daty.
            benchmark_returns: Series log-stóp zwrotu benchmarku.
                               Index: daty (może nie pokrywać się w pełni).

        Returns:
            Krotka (mu_capm, cov_hist):
                mu_capm: ndarray (n_assets,) – dzienne stopy zwrotu wg CAPM.
                cov_hist: ndarray (n_assets, n_assets) – historyczna macierz kowariancji.
        """
        # Wyrównaj indeksy: weź tylko daty wspólne dla spółek i benchmarku
        aligned_bench: pd.Series = benchmark_returns.reindex(train_returns.index).dropna()
        aligned_assets: pd.DataFrame = train_returns.reindex(aligned_bench.index)

        # Historyczna dzienna stopa zwrotu rynku (prosta średnia)
        mu_m: float = float(aligned_bench.mean())

        # Premia za ryzyko rynkowe (dzienna): E[r_m] - r_f
        # Historycznie ~5-7% rocznie; w przeliczeniu na dzień ~0.02-0.028%
        market_risk_premium: float = mu_m - self.risk_free

        # Wariancja rynku (dzienna): Var(r_m)
        var_m: float = float(aligned_bench.var())

        # Beta każdej spółki: β_i = Cov(r_i, r_m) / Var(r_m)
        # Cov(r_i, r_m): kowariancja spółki z rynkiem
        # Im wyższa β, tym spółka silniej reaguje na zmiany indeksu
        betas: np.ndarray = np.array([
            float(aligned_assets[col].cov(aligned_bench)) / (var_m + 1e-12)
            for col in aligned_assets.columns
        ])

        # Oczekiwana stopa zwrotu wg CAPM: r_f + β * (E[r_m] - r_f)
        mu_capm: np.ndarray = self.risk_free + betas * market_risk_premium

        # Macierz kowariancji z danych historycznych (ta sama co w klasycznym MVO)
        # Próbkowa macierz kowariancji jest symetryczna i dodatnio półokreślona
        cov_hist: np.ndarray = aligned_assets.cov().values

        # Wydruk informacyjny
        asset_names = list(train_returns.columns)
        print(f"\n  Premia za ryzyko rynkowe (dzienna): {market_risk_premium:.6f} "
              f"({market_risk_premium * 252:.2%} rocznie)")
        print(f"  Beta spółek:")
        for name, beta in zip(asset_names, betas):
            bar = "█" * int(abs(beta) * 10) + ("+" if beta > 0 else "-")
            print(f"    {name:6}: β = {beta:+.3f}  {bar}")

        return mu_capm, cov_hist

    @staticmethod
    def compute_benchmark_returns(
        benchmark_prices: pd.Series,
    ) -> pd.Series:
        """Oblicza log-stopy zwrotu benchmarku z jego cen.

        Args:
            benchmark_prices: Series dziennych cen zamknięcia benchmarku.

        Returns:
            Series dziennych log-stóp zwrotu benchmarku.
        """
        # Taka sama formuła jak dla spółek: r_t = ln(P_t / P_{t-1})
        bench_ret: pd.Series = np.log(
            benchmark_prices / benchmark_prices.shift(1)
        ).dropna()
        return bench_ret