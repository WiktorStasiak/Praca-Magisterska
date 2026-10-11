"""
visualization/plots.py – Wizualizacja wyników optymalizacji portfela
=====================================================================
Moduł zawiera wszystkie wykresy generowane przez program:
  1. Krzywa uczenia LSTM (train vs. validation loss)
  2. Rozkłady prognoz Monte Carlo Dropout
  3. Porównanie wag portfeli (słupkowy)
  4. Macierz korelacji historycznej (heatmapa)
  5. Skumulowane stopy zwrotu – główny wykres backtestingu
  6. Rolling Sharpe (30-dniowy) – stabilność strategii w czasie
  7. Analiza drawdown – obsunięcia kapitału

Wszystkie wykresy są zapisywane do pliku PNG w katalogu roboczym
oraz wyświetlane na ekranie (plt.show()).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# Globalny styl wykresów – spójny wygląd w całym projekcie
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.2,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Paleta kolorów dla 3 strategii + benchmark – wyraziste i rozróżnialne
COLORS = {
    "hybrid":    "#1565C0",   # ciemny niebieski – LSTM + MC Dropout
    "capm":      "#E53935",   # ciemnoczerwony   – Klasyczny CAPM
    "classic":   "#6A1B9A",   # fioletowy         – Klasyczny historyczny
    "benchmark": "#2E7D32",   # ciemnozielony     – DJIA benchmark
    "equal":     "#F57F17",   # pomarańczowy      – 1/N równoważony
}

OUTPUT_DIR = Path(".")   # katalog wyjściowy dla plików PNG


class Plotter:
    """Generuje i zapisuje wszystkie wykresy wyników programu.

    Attributes:
        output_dir: Katalog, w którym zapisywane są pliki PNG.
    """

    def __init__(self, output_dir: str = ".") -> None:
        """Inicjalizuje Plotter z katalogiem wyjściowym.

        Args:
            output_dir: Ścieżka katalogu do zapisu wykresów.
                        Domyślnie bieżący katalog roboczy.
        """
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig: plt.Figure, filename: str) -> None:
        """Zapisuje rysunek do pliku PNG i wyświetla komunikat.

        Args:
            fig: Obiekt Figure matplotlib do zapisu.
            filename: Nazwa pliku (lub ścieżka względna, z rozszerzeniem .png).
        """
        path = self.output_dir / filename

        # Automatycznie twórz katalog (wraz z podkatalogami, jeśli nie istnieją):
        path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # Opcjonalnie: zwalnia pamięć po zapisie wykresu
        print(f"  ✓ Zapisano: {path}")

    # ── Wykres 1: Krzywa uczenia LSTM ──────────────────────────────────

    def plot_training_history(
        self,
        history: "tf.keras.callbacks.History",
    ) -> None:
        """Rysuje krzywe MSE treningu i walidacji w funkcji epok.

        Krzywa uczenia pozwala ocenić:
          • Czy model konwerguje (maleje loss)?
          • Czy nie ma overfittingu (val_loss ≫ train_loss)?
          • Kiedy EarlyStopping zatrzymał trening (pionowa linia)?

        Args:
            history: Obiekt History zwrócony przez model.fit().
        """
        train_loss: list[float] = history.history["loss"]
        val_loss:   list[float] = history.history["val_loss"]
        epochs:     list[int]   = list(range(1, len(train_loss) + 1))

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(epochs, train_loss, color=COLORS["hybrid"],
                linewidth=2.0, label="Strata treningowa (MSE)")
        ax.plot(epochs, val_loss,   color=COLORS["capm"],
                linewidth=2.0, linestyle="--", label="Strata walidacyjna (MSE)")

        # Zaznacz epokę z najniższą val_loss (gdzie EarlyStopping wybrał wagi)
        best_epoch: int = int(np.argmin(val_loss)) + 1
        ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5,
                   label=f"Najlepsza epoka ({best_epoch})")

        ax.set_title("Krzywa uczenia modelu LSTM + MC Dropout",
                     fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Epoka")
        ax.set_ylabel("Błąd MSE (Mean Squared Error)")
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))

        plt.tight_layout()
        self._save(fig, "plots/01_lstm_learning_curve.png")
        plt.show()

    # ── Wykres 2: Rozkłady MC Dropout ──────────────────────────────────

    def plot_mc_distributions(
        self,
        mc_returns:  np.ndarray,
        asset_names: list[str],
        max_assets:  int = 10,
    ) -> None:
        """Rysuje histogramy rozkładów prognoz MC Dropout dla wybranych spółek.

        Każdy histogram pokazuje rozkład S=100 prognozowanych stóp zwrotu
        dla jednej spółki. Szerokość rozkładu = niepewność modelu:
          • Wąski → model pewny prognozy
          • Szeroki → model niepewny (duże ryzyko/zmienność)

        Dla czytelności przy 30 spółkach DJIA wyświetlamy max_assets
        o największej niepewności (największe σ_MC).

        Args:
            mc_returns: ndarray (n_samples, n_assets) – próbki MC Dropout.
            asset_names: Lista nazw spółek odpowiadających kolumnom.
            max_assets: Maksymalna liczba wyświetlanych spółek.
        """
        n_all: int = len(asset_names)

        # Wybierz spółki o największej niepewności MC (σ_MC)
        sigma_mc: np.ndarray = mc_returns.std(axis=0)
        top_idx:  np.ndarray = np.argsort(sigma_mc)[::-1][:max_assets]
        selected_names:    list[str]   = [asset_names[i] for i in top_idx]
        selected_returns:  np.ndarray  = mc_returns[:, top_idx]

        n: int = len(selected_names)
        ncols: int = min(5, n)
        nrows: int = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.5 * nrows),
                                 squeeze=False)

        colors = plt.cm.tab10(np.linspace(0, 1, n))

        for idx, (name, ax) in enumerate(
            zip(selected_names, axes.flat)
        ):
            col_returns: np.ndarray = selected_returns[:, idx]
            mean_r:  float = float(col_returns.mean())
            std_r:   float = float(col_returns.std())

            ax.hist(col_returns, bins=25, color=colors[idx],
                    edgecolor="white", alpha=0.85, density=True)
            ax.axvline(mean_r, color="black", linewidth=1.8,
                       linestyle="--", label=f"μ={mean_r:.4f}")
            ax.axvline(mean_r + std_r, color="gray", linewidth=1.0,
                       linestyle=":", alpha=0.7)
            ax.axvline(mean_r - std_r, color="gray", linewidth=1.0,
                       linestyle=":", alpha=0.7, label=f"σ={std_r:.4f}")

            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Dzienna stopa zwrotu", fontsize=8)
            ax.set_ylabel("Gęstość", fontsize=8)
            ax.legend(fontsize=7)

        # Ukryj puste subploty (gdy n < nrows*ncols)
        for ax in axes.flat[n:]:
            ax.set_visible(False)

        fig.suptitle(
            f"Rozkłady prognoz MC Dropout – Top {n} spółek wg niepewności modelu\n"
            f"(z {n_all} spółek DJIA; szerszy rozkład = większa niepewność LSTM)",
            fontsize=13, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        self._save(fig, "plots/02_mc_dropout_distributions.png")
        plt.show()

    # ── Wykres 3: Wagi portfeli ─────────────────────────────────────────

    def plot_portfolio_weights(
        self,
        weights_dict: dict[str, np.ndarray],
        asset_names:  list[str],
    ) -> None:
        """Rysuje wykres słupkowy wag dla wszystkich optymalizowanych portfeli.

        Porównanie wag pozwala ocenić:
          • Jak bardzo strategie różnią się od siebie?
          • Które spółki są preferowane przez LSTM vs. CAPM?
          • Czy portfele są dobrze zdywersyfikowane?

        Args:
            weights_dict: Słownik {etykieta: wektor_wag}.
                          Przykład: {'Hybrydowy': w1, 'CAPM': w2}.
            asset_names: Lista nazw spółek (oś X).
        """
        n_assets:     int = len(asset_names)
        n_strategies: int = len(weights_dict)

        fig, ax = plt.subplots(figsize=(max(16, n_assets * 0.7), 6))

        # Szerokość i pozycje słupków
        bar_width: float      = 0.8 / n_strategies
        x:         np.ndarray = np.arange(n_assets)

        strategy_colors = [COLORS["hybrid"], COLORS["capm"],
                           COLORS["classic"], COLORS["equal"]]

        for i, (label, weights) in enumerate(weights_dict.items()):
            offset: float = (i - (n_strategies - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, weights,
                width=bar_width * 0.9,
                label=label,
                color=strategy_colors[i % len(strategy_colors)],
                alpha=0.85,
                edgecolor="white",
            )
            # Etykiety na słupkach > 5%
            for bar in bars:
                h = bar.get_height()
                if h > 0.05:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.005,
                        f"{h:.0%}",
                        ha="center", va="bottom",
                        fontsize=7, fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(asset_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Waga portfela")
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_title(
            "Optymalne wagi portfela – porównanie strategii inwestycyjnych",
            fontsize=14, fontweight="bold", pad=12,
        )
        ax.legend(fontsize=10, loc="upper right")

        # Pozioma linia = waga równoważona (1/N)
        equal_w: float = 1.0 / n_assets
        ax.axhline(equal_w, color=COLORS["equal"], linewidth=1.2,
                   linestyle=":", alpha=0.7,
                   label=f"Waga równoważona (1/N = {equal_w:.1%})")

        plt.tight_layout()
        self._save(fig, "plots/03_portfolio_weights.png")
        plt.show()

    # ── Wykres 4: Macierz korelacji ─────────────────────────────────────

    def plot_correlation_matrix(
        self,
        returns:     pd.DataFrame,
        title:       str = "Historyczna macierz korelacji spółek DJIA",
    ) -> None:
        """Rysuje heatmapę historycznej macierzy korelacji Pearsona.

        Korelacja Pearsona ρ_{i,j} ∈ [-1, 1]:
          ρ > 0  → spółki rosną/spadają razem (dywersyfikacja słabsza)
          ρ < 0  → spółki poruszają się przeciwnie (dywersyfikacja silna)
          ρ = 0  → brak liniowej zależności

        W portfelu chcemy mieszanki spółek z niską wzajemną korelacją,
        bo zmniejsza to całkowite ryzyko portfela (Σ w' Σ w).

        Args:
            returns: DataFrame log-stóp zwrotu spółek.
            title: Tytuł wykresu.
        """
        corr: pd.DataFrame = returns.corr()
        n:    int           = len(corr)

        fig, ax = plt.subplots(figsize=(max(12, n * 0.45), max(10, n * 0.4)))

        # Kolorowa mapa: niebieski (ρ=-1) → biały (ρ=0) → czerwony (ρ=+1)
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="auto")

        # Oś X i Y: nazwy spółek
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)

        # Wartości w komórkach (tylko gdy < 30 spółek – czytelność)
        if n <= 30:
            for i in range(n):
                for j in range(n):
                    val = corr.values[i, j]
                    color = "white" if abs(val) > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color=color)

        plt.colorbar(im, ax=ax, label="Korelacja Pearsona ρ",
                     fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

        plt.tight_layout()
        self._save(fig, "plots/04_correlation_matrix.png")
        plt.show()

    # ── Wykres 5: Skumulowane zwroty (główny) ──────────────────────────

    def plot_cumulative_returns(
        self,
        portfolios: dict[str, pd.Series],
    ) -> None:
        """Rysuje główny wykres backtestingu – skumulowane stopy zwrotu.

        To kluczowy wykres projektu: pokazuje, jak zachowywały się
        wszystkie strategie w tym samym okresie testowym (2024–2025).

        Linie powyżej 0% = strategia jest na plusie.
        Linia wyżej na końcu = lepsza stopa zwrotu łącznie.

        Args:
            portfolios: Słownik {nazwa: Series skumulowanych zwrotów}.
                        Kluczowe klucze: 'hybrid', 'capm', 'classic',
                        'benchmark', 'equal' (muszą odpowiadać kolorom).
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        color_map: dict[str, str] = {
            **COLORS,
            # Fallback: szary dla nieznanych kluczy
        }
        line_styles: dict[str, str] = {
            "hybrid":    "-",
            "capm":      "--",
            "classic":   "-.",
            "benchmark": ":",
            "equal":     (0, (3, 1, 1, 1)),   # punktowo-kreślony
        }

        for key, series in portfolios.items():
            color     = color_map.get(key, "#555555")
            linestyle = line_styles.get(key, "-")
            lw        = 2.5 if key == "hybrid" else 2.0

            ax.plot(
                series.index,
                series.values * 100,   # przeliczamy na procenty
                label=series.name,
                color=color,
                linewidth=lw,
                linestyle=linestyle,
            )

            # Anotacja wyniku końcowego przy ostatnim punkcie
            final_val: float = float(series.iloc[-1] * 100)
            ax.annotate(
                f"{final_val:+.1f}%",
                xy=(series.index[-1], final_val),
                xytext=(8, 0), textcoords="offset points",
                fontsize=9, fontweight="bold",
                color=color, va="center",
            )

        # Linia zero – punkt odniesienia (wynik inwestycji początkowej)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")

        ax.set_title(
            "Backtesting: Skumulowane stopy zwrotu (out-of-sample, 2024–2025)",
            fontsize=15, fontweight="bold", pad=14,
        )
        ax.set_xlabel("Data", fontsize=11)
        ax.set_ylabel("Skumulowany zwrot [%]", fontsize=11)

        # Format osi X: miesiąc i rok
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.legend(fontsize=10, loc="upper left")

        plt.tight_layout()
        self._save(fig, "plots/05_cumulative_returns.png")
        plt.show()

    # ── Wykres 6: Rolling Sharpe ────────────────────────────────────────

    def plot_rolling_sharpe(
        self,
        portfolios_daily: dict[str, pd.Series],
        window:           int   = 63,
        risk_free_daily:  float = 0.045 / 252,
    ) -> None:
        """Rysuje kroczący (rolling) wskaźnik Sharpe'a w czasie.

        Rolling Sharpe z oknem = 63 sesje (≈ 3 miesiące):
            SR_t = (μ_{t-w:t} - r_f) / σ_{t-w:t} * √252

        Interpretacja:
          SR > 1.0  → strategia generuje dobry zwrot ważony ryzykiem
          SR < 0.0  → strategia traci relatywnie do stopy wolnej
          Stabilny SR → przewidywalna strategia

        Wykres pokazuje, czy przewaga jednej strategii nad drugą
        jest konsekwentna, czy tylko wynika z kilku dobrych miesięcy.

        Args:
            portfolios_daily: Słownik {nazwa: Series dziennych log-zwrotów}.
            window: Długość okna kroczącego w sesjach giełdowych.
            risk_free_daily: Dzienna stopa wolna od ryzyka.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        color_map = COLORS
        line_styles = {"hybrid": "-", "capm": "--",
                       "classic": "-.", "benchmark": ":", "equal": (0,(3,1,1,1))}

        rf_annual: float = risk_free_daily * 252   # annualizowana r_f

        for key, daily_ret in portfolios_daily.items():
            # Kroczące annualizowane statystyki
            roll_mean: pd.Series = daily_ret.rolling(window).mean() * 252
            roll_std:  pd.Series = daily_ret.rolling(window).std()  * np.sqrt(252)

            # Rolling Sharpe: (μ_roczna - r_f) / σ_roczna
            roll_sharpe: pd.Series = (roll_mean - rf_annual) / (roll_std + 1e-10)

            ax.plot(
                roll_sharpe.index,
                roll_sharpe.values,
                label=daily_ret.name if hasattr(daily_ret, "name") else key,
                color=color_map.get(key, "#555555"),
                linewidth=2.0,
                linestyle=line_styles.get(key, "-"),
            )

        ax.axhline(0,   color="gray",  linewidth=0.8, linestyle="-")
        ax.axhline(1.0, color="green", linewidth=0.8, linestyle=":",
                   alpha=0.5, label="SR = 1.0 (dobry)")

        ax.set_title(
            f"Kroczący wskaźnik Sharpe'a (okno = {window} sesji ≈ {window//21} mies.)",
            fontsize=14, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Data")
        ax.set_ylabel("Sharpe Ratio (roczny)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        ax.legend(fontsize=10)

        plt.tight_layout()
        self._save(fig, "plots/06_rolling_sharpe.png")
        plt.show()

    # ── Wykres 7: Analiza Drawdown ──────────────────────────────────────

    def plot_drawdown(
        self,
        portfolios: dict[str, pd.Series],
    ) -> None:
        """Rysuje wykresy obsunięcia kapitału (drawdown) dla każdej strategii.

        Drawdown w dniu t:
            DD(t) = (Wealth(t) - max_{s≤t} Wealth(s)) / max_{s≤t} Wealth(s)

        Zawsze ≤ 0. Głębszy (bardziej ujemny) DD = większe straty od szczytu.

        Analiza drawdown jest kluczowa dla psychologii inwestowania:
        inwestor, który doświadczył DD = -40%, musi zarobić +67% aby
        wrócić do poprzedniego szczytu (efekt asymetrii strat/zysków).

        Args:
            portfolios: Słownik {klucz: Series skumulowanych zwrotów}.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        color_map = COLORS
        line_styles = {"hybrid": "-", "capm": "--",
                       "classic": "-.", "benchmark": ":", "equal": (0,(3,1,1,1))}

        for key, cum_ret in portfolios.items():
            wealth: pd.Series     = 1.0 + cum_ret
            running_max: pd.Series = wealth.cummax()
            drawdown: pd.Series    = (wealth / running_max - 1.0) * 100  # w %

            ax.fill_between(
                drawdown.index,
                drawdown.values,
                0,
                alpha=0.20,
                color=color_map.get(key, "#555555"),
            )
            ax.plot(
                drawdown.index,
                drawdown.values,
                label=cum_ret.name,
                color=color_map.get(key, "#555555"),
                linewidth=1.8,
                linestyle=line_styles.get(key, "-"),
            )

        ax.set_title(
            "Analiza obsunięcia kapitału (Drawdown) – okres testowy",
            fontsize=14, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Data")
        ax.set_ylabel("Drawdown [%]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.legend(fontsize=10)

        plt.tight_layout()
        self._save(fig, "plots/07_drawdown_analysis.png")
        plt.show()