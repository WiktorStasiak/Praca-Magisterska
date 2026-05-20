"""
data/loader.py – Pobieranie i przygotowanie danych rynkowych
=============================================================
Moduł odpowiedzialny za:
  • Pobieranie cen zamknięcia z Yahoo Finance (yfinance)
  • Obliczanie logarytmicznych stóp zwrotu
  • Podział danych na zbiory: treningowy, walidacyjny i testowy
  • Skalowanie danych dla sieci LSTM (MinMaxScaler)
  • Budowanie sekwencji czasowych (tenzor wejściowy LSTM)
"""

from __future__ import annotations

import warnings
from typing import Optional
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    """Pobiera i przygotowuje dane cen akcji z Yahoo Finance.

    Klasa obsługuje pobranie danych dla wielu tickerów,
    wyrównanie indeksów czasowych (wspólne daty sesji)
    oraz obliczenie logarytmicznych stóp zwrotu.

    Logarytmiczna stopa zwrotu w dniu t:
        r_t = ln(P_t / P_{t-1})

    Właściwości logarytmicznych stóp zwrotu:
      • addytywność w czasie: r_{t→t+n} = Σ r_i
      • lepsza normalność rozkładu niż stopy proste
      • symetria: zysk 100% ↔ strata 50% mają tę samą wartość |r|

    Attributes:
        tickers: Słownik {nazwa: ticker_yf} dla pobieranych spółek.
        start_date: Data początkowa w formacie 'YYYY-MM-DD'.
        end_date: Data końcowa w formacie 'YYYY-MM-DD'.
        prices: DataFrame dziennych cen zamknięcia (po pobraniu).
        returns: DataFrame dziennych log-stóp zwrotu (po pobraniu).
    """

    def __init__(
        self,
        tickers: dict[str, str],
        start_date: str,
        end_date: str,
    ) -> None:
        """Inicjalizuje DataLoader z parametrami pobierania.

        Args:
            tickers: Słownik mapujący nazwę spółki na ticker Yahoo Finance.
                     Przykład: {'AAPL': 'AAPL', 'MSFT': 'MSFT'}
            start_date: Data początkowa pobierania, format 'YYYY-MM-DD'.
            end_date: Data końcowa pobierania, format 'YYYY-MM-DD'.
        """
        self.tickers:    dict[str, str]       = tickers
        self.start_date: str                  = start_date
        self.end_date:   str                  = end_date
        self.prices:     Optional[pd.DataFrame] = None
        self.returns:    Optional[pd.DataFrame] = None

    # ── Pobieranie danych ──────────────────────────────────────────────

    # ── Wewnętrzna metoda pomocnicza ──────────────────────────────────

    @staticmethod
    def _yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
        """Wywołuje yf.download() w sposób kompatybilny z wieloma wersjami yfinance.

        Zmiany API yfinance między wersjami:
          • < 0.2.x : kolumny jednopoziomowe, parametr show_errors dostępny
          • ≥ 0.2.x : kolumny wielopoziomowe (MultiIndex) gdy group_by='ticker'
          • ≥ 1.0.x : show_errors usunięty; dodany multi_level_index=False
                       który wymusza kolumny jednopoziomowe dla jednego tickera

        Strategia: próbujemy najpierw z multi_level_index=False (yfinance ≥ 1.x),
        a w razie błędu TypeError (starsza wersja) wołamy bez tego parametru.

        Args:
            ticker: Symbol tickera Yahoo Finance (np. 'AAPL').
            start:  Data początkowa 'YYYY-MM-DD'.
            end:    Data końcowa 'YYYY-MM-DD'.

        Returns:
            DataFrame z cenami; kolumny jednopoziomowe (bez MultiIndex).
        """
        # Wspólne parametry dla wszystkich wersji yfinance
        common_kwargs = dict(
            start=start,
            end=end,
            auto_adjust=True,   # ceny korygowane o splity i dywidendy
            progress=False,     # wyłącz pasek postępu w konsoli
        )

        try:
            # yfinance ≥ 1.0: multi_level_index=False → kolumny jednopoziomowe
            # dzięki temu raw["Close"] od razu zwraca Series, nie DataFrame
            raw = yf.download(ticker, multi_level_index=False, **common_kwargs)
        except TypeError:
            # yfinance < 1.0: parametr multi_level_index nie istnieje
            raw = yf.download(ticker, **common_kwargs)

        return raw

    def download_prices(self) -> pd.DataFrame:
        """Pobiera dzienne kursy zamknięcia dla wszystkich tickerów.

        Dla każdej spółki pobieramy dane z Yahoo Finance. Kursy są
        korygowane o splity i dywidendy (auto_adjust=True), dzięki
        czemu odzwierciedlają rzeczywistą stopę zwrotu inwestora.

        Po pobraniu wyrównujemy DataFrame do wspólnych dat sesji
        (dropna usuwa dni, gdy choć jedna spółka nie miała notowań).

        Returns:
            DataFrame z dziennymi cenami zamknięcia.
            Index: pd.DatetimeIndex (daty sesji giełdowych).
            Columns: nazwy spółek (klucze ze słownika tickers).

        Raises:
            RuntimeError: Jeśli udało się pobrać dane dla mniej niż
                          5 spółek (zbyt mało do optymalizacji portfela).
        """
        print("─" * 60)
        print("► Pobieranie danych z Yahoo Finance...")
        print(f"  yfinance wersja: {yf.__version__}")

        frames: dict[str, pd.Series] = {}

        for name, tkr in self.tickers.items():
            try:
                raw = self._yf_download(tkr, self.start_date, self.end_date)

                if raw.empty:
                    print(f"  [BRAK] {name} ({tkr}): brak danych")
                    continue

                # Wyodrębniamy kolumnę 'Close'.
                # W yfinance ≥ 1.x z multi_level_index=False jest to zwykła
                # kolumna 'Close'. W starszych wersjach może być MultiIndex
                # ('Close', ticker) – squeeze() spłaszcza do Series.
                close_col = raw["Close"]
                if isinstance(close_col, pd.DataFrame):
                    # MultiIndex: weź pierwszą (i jedyną) kolumnę
                    close_col = close_col.iloc[:, 0]

                close: pd.Series = close_col.dropna()

                if len(close) < 100:
                    print(f"  [SKIP] {name} ({tkr}): tylko {len(close)} sesji")
                    continue

                frames[name] = close
                print(f"  ✓ {name} ({tkr}): {len(close)} sesji  "
                      f"[{close.index[0].date()} → {close.index[-1].date()}]")

            except Exception as exc:
                print(f"  [ERR] {name} ({tkr}): {str(exc)[:80]}")

        # ── Zbuduj wspólny DataFrame ───────────────────────────────────
        if len(frames) < 5:
            raise RuntimeError(
                f"Pobrano dane tylko dla {len(frames)} spółek. "
                "Sprawdź połączenie z internetem lub listę tickerów."
            )

        # Łączymy wszystkie Serie w jeden DataFrame;
        # tam gdzie brakuje danych pojawia się NaN
        df_raw: pd.DataFrame = pd.DataFrame(frames)

        # dropna() usuwa dni, gdy choć jedna spółka nie ma notowań
        # (np. różne święda giełdowe lub nowa spółka w indeksie)
        df: pd.DataFrame = df_raw.dropna()

        print(f"\n  ✓ Łącznie: {df.shape[1]} spółek, {len(df)} wspólnych sesji")
        print(f"  ✓ Okres: {df.index[0].date()} → {df.index[-1].date()}")

        self.prices = df
        return df

    def download_benchmark(self, ticker: str) -> pd.Series:
        """Pobiera ceny benchmarku (np. ^DJI – Dow Jones Industrial Average).

        Benchmark służy do porównania wyników portfela z "rynkiem" –
        strategią pasywną polegającą na trzymaniu całego indeksu.

        Args:
            ticker: Ticker benchmarku w Yahoo Finance (np. '^DJI').

        Returns:
            Series z dziennymi cenami zamknięcia benchmarku.
            Pusta Series jeśli pobranie się nie powiodło.
        """
        try:
            raw = self._yf_download(ticker, self.start_date, self.end_date)

            close_col = raw["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]

            benchmark = close_col.dropna()
            print(f"\n  ✓ Benchmark {ticker}: {len(benchmark)} sesji")
            return benchmark

        except Exception as exc:
            print(f"  [WARN] Benchmark {ticker} niedostępny: {str(exc)[:60]}")
            # Zwracamy pustą serię – caller obsłuży brak benchmarku
            return pd.Series(dtype=float)

    # ── Obliczanie stóp zwrotu ─────────────────────────────────────────

    def compute_returns(self, prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Oblicza dzienne logarytmiczne stopy zwrotu.

        Wzór: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

        Implementacja z użyciem pandas jest równoważna:
            returns = np.log(prices / prices.shift(1)).dropna()

        Args:
            prices: DataFrame cen. Jeśli None, używa self.prices.

        Returns:
            DataFrame log-stóp zwrotu. Pierwszy wiersz jest usunięty
            (brak poprzedniej ceny dla t=0), więc len(returns) = len(prices) - 1.
        """
        if prices is None:
            prices = self.prices

        if prices is None:
            raise RuntimeError("Najpierw wywołaj download_prices().")

        # np.log(P_t / P_{t-1}) = logarytmiczna stopa zwrotu dnia t
        returns: pd.DataFrame = np.log(prices / prices.shift(1)).dropna()
        self.returns = returns
        return returns

    # ── Podział danych ──────────────────────────────────────────────────

    def split_data(
        self,
        prices:       pd.DataFrame,
        returns:      pd.DataFrame,
        train_start:  str,
        train_end:    str,
        val_start:    str,
        val_end:      str,
        test_start:   str,
        test_end:     str,
    ) -> dict[str, pd.DataFrame]:
        """Dzieli dane na trzy rozłączne okresy: trening, walidacja, test.

        Schemat podziału stosowany w tym projekcie:
            TRENING   2011-01-01 → 2021-12-31  (11 lat)
            WALIDACJA 2022-01-01 → 2023-12-31  ( 2 lata)
            TEST      2024-01-01 → 2025-12-31  ( 2 lata)

        Ważne: Dane testowe NIE są używane podczas budowania modelu –
        ani do treningu, ani do walidacji. Zapobiega to data snooping bias
        (nieświadomemu dostosowaniu modelu do danych testowych).

        Args:
            prices: DataFrame cen zamknięcia.
            returns: DataFrame log-stóp zwrotu.
            train_start: Początek okresu treningowego.
            train_end: Koniec okresu treningowego.
            val_start: Początek okresu walidacyjnego.
            val_end: Koniec okresu walidacyjnego.
            test_start: Początek okresu testowego.
            test_end: Koniec okresu testowego.

        Returns:
            Słownik z kluczami: 'train_prices', 'train_returns',
            'val_prices', 'val_returns', 'test_prices', 'test_returns'.
        """
        def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
            """Filtruje DataFrame do podanego zakresu dat."""
            return df.loc[start:end]

        # Wycinamy każdy okres ze słownika przez zakres dat indeksu
        split: dict[str, pd.DataFrame] = {
            "train_prices":  _slice(prices,  train_start, train_end),
            "train_returns": _slice(returns, train_start, train_end),
            "val_prices":    _slice(prices,  val_start,   val_end),
            "val_returns":   _slice(returns, val_start,   val_end),
            "test_prices":   _slice(prices,  test_start,  test_end),
            "test_returns":  _slice(returns, test_start,  test_end),
        }

        # Wydruk informacyjny o rozmiarach zbiorów
        for period in ["train", "val", "test"]:
            p = split[f"{period}_prices"]
            print(f"  {period.capitalize():10}: "
                  f"{p.index[0].date()} → {p.index[-1].date()} "
                  f"({len(p)} sesji)")

        return split


class SequenceBuilder:
    """Przygotowuje sekwencje czasowe i skalowanie dla modelu LSTM.

    LSTM wymaga danych wejściowych w formacie 3D:
        (n_samples, lookback, n_features)

    gdzie:
        n_samples  = liczba próbek (okien) w zbiorze
        lookback   = długość okna historycznego (np. 60 dni)
        n_features = liczba aktywów (np. 30 spółek DJIA)

    Skalowanie (MinMaxScaler) przekształca ceny do zakresu [0, 1],
    co przyspiesza zbieżność optymalizatora Adam i zapobiega dominacji
    spółek o wysokich cenach nominalnych (np. NVDA vs. KO).

    Attributes:
        lookback: Długość okna wejściowego w dniach.
        scaler: Dopasowany obiekt MinMaxScaler (do odwrócenia normalizacji).
    """

    def __init__(self, lookback: int) -> None:
        """Inicjalizuje SequenceBuilder.

        Args:
            lookback: Liczba dni historycznych w jednym oknie wejściowym.
                      Na przykład lookback=60 oznacza, że model przewiduje
                      dzień t na podstawie cen z dni t-60, ..., t-1.
        """
        self.lookback: int                    = lookback
        self.scaler:   Optional[MinMaxScaler] = None

    def fit_scaler(self, train_prices: pd.DataFrame) -> MinMaxScaler:
        """Dopasowuje MinMaxScaler do cen treningowych.

        Scaler jest dopasowywany WYŁĄCZNIE na danych treningowych,
        a nie na całym zbiorze. Zastosowanie scalera dopasowanego
        do całości danych (data leakage) zawyżałoby wyniki.

        MinMaxScaler skaluje każdą kolumnę (spółkę) niezależnie:
            x_scaled = (x - x_min) / (x_max - x_min)
        gdzie x_min, x_max są minimalnymi i maksymalnymi cenami
        WYŁĄCZNIE w zbiorze treningowym.

        Args:
            train_prices: DataFrame cen z okresu treningowego.

        Returns:
            Dopasowany obiekt MinMaxScaler (zapamiętany w self.scaler).
        """
        # feature_range=(0,1): ceny zostaną przekształcone do [0, 1]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(train_prices.values)
        return self.scaler

    def build_sequences(
        self,
        prices:  pd.DataFrame,
        fit:     bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Skaluje ceny i buduje sekwencje (X, y) dla LSTM.

        Dla każdego indeksu i w przedziale [lookback, N):
            X[i] = scaled_prices[i-lookback : i]   # (lookback, n_features)
            y[i] = scaled_prices[i]                 # (n_features,)

        czyli model uczy się: "mając ceny z ostatnich `lookback` dni,
        przewidź ceny jutra".

        Args:
            prices: DataFrame cen do przetworzenia.
            fit: Jeśli True, scaler zostaje dopasowany do tych cen
                 (używaj TYLKO dla zbioru treningowego!).

        Returns:
            Krotka (X, y):
                X: ndarray kształtu (n_samples, lookback, n_features)
                y: ndarray kształtu (n_samples, n_features)

        Raises:
            RuntimeError: Jeśli scaler nie był wcześniej dopasowany
                          i fit=False.
        """
        if fit:
            # Dopasowanie scalera na podanych cenach (tylko dla train!)
            self.fit_scaler(prices)

        if self.scaler is None:
            raise RuntimeError(
                "Scaler nie jest dopasowany. Wywołaj fit_scaler() "
                "lub użyj build_sequences(prices, fit=True)."
            )

        # Skalujemy ceny do zakresu [0, 1] (każda spółka osobno)
        scaled: np.ndarray = self.scaler.transform(prices.values)

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        # Przesuwamy okno po czasie, tworząc próbki (X[i], y[i])
        for i in range(self.lookback, len(scaled)):
            # Okno historyczne: lookback dni wstecz
            X_list.append(scaled[i - self.lookback : i, :])
            # Target: ceny następnego dnia
            y_list.append(scaled[i, :])

        X: np.ndarray = np.array(X_list)  # (n_samples, lookback, n_features)
        y: np.ndarray = np.array(y_list)  # (n_samples, n_features)

        return X, y