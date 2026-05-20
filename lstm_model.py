"""
models/lstm_model.py – Model LSTM z Monte Carlo Dropout
=========================================================
Moduł zawiera:
  • LSTMForecaster – budowanie, trening i prognozowanie modelu LSTM
  • mc_dropout_predict() – próbkowanie Monte Carlo z aktywnym Dropout

Kluczowa idea: Monte Carlo Dropout
────────────────────────────────────
Standardowy Dropout jest aktywny TYLKO podczas treningu.
Gal & Ghahramani (2016) wykazali, że włączenie Dropout podczas
inferencji (flaga training=True) jest równoważne aproksymacji
bayesowskiej sieci neuronowej.

Poprzez wielokrotne uruchomienie modelu z losowym Dropout
(MC_SAMPLES razy) uzyskujemy rozkład prognoz. Jego:
  • Średnia  → punkt prognozowy (oczekiwana stopa zwrotu)
  • Wariancja → miara niepewności epistemicznej (niewiedzowej)

Referencja:
  Gal, Y., Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
  Representing Model Uncertainty in Deep Learning. ICML 2016.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping, History
from sklearn.preprocessing import MinMaxScaler


class LSTMForecaster:
    """Dwuwarstwowy model LSTM z Dropout do prognozowania cen akcji.

    Architektura sieci:
        Input(lookback, n_features)
            → LSTM(units, return_sequences=True)   # warstwa 1: wyodrębnia cechy
            → Dropout(rate)                         # regularyzacja
            → LSTM(units, return_sequences=False)   # warstwa 2: koduje sekwencję
            → Dropout(rate)                         # regularyzacja / MC Dropout
            → Dense(n_features, activation='linear')# prognozy cen

    Dlaczego dwie warstwy LSTM?
      • Pierwsza warstwa uczy się niskopoziomowych wzorców
        (krótkoterminowe trendy, dzienne oscylacje).
      • Druga warstwa łączy je w wyższopoziomowe reprezentacje
        (trendy wielomiesięczne, zależności między spółkami).

    Attributes:
        n_features: Liczba aktywów w portfelu (spółki DJIA).
        lookback: Długość okna wejściowego w dniach.
        lstm_units: Liczba komórek pamięci w każdej warstwie LSTM.
        dropout_rate: Współczynnik Dropout (0.0 – 1.0).
        model: Skompilowany model Keras (po wywołaniu build()).
    """

    def __init__(
        self,
        n_features:   int,
        lookback:     int,
        lstm_units:   int   = 128,
        dropout_rate: float = 0.20,
    ) -> None:
        """Inicjalizuje parametry modelu LSTM.

        Args:
            n_features: Liczba aktywów (== liczba kolumn w danych wejściowych).
            lookback: Liczba dni w oknie historycznym (LSTM context window).
            lstm_units: Liczba neuronów w każdej warstwie LSTM.
                        Wyższa wartość = większa pojemność modelu,
                        ale też dłuższy czas treningu.
            dropout_rate: Frakcja neuronów losowo zerowanych w Dropout.
                          Typowe wartości: 0.1 – 0.5.
        """
        self.n_features:   int   = n_features
        self.lookback:     int   = lookback
        self.lstm_units:   int   = lstm_units
        self.dropout_rate: float = dropout_rate
        self.model:        Optional[Model] = None

    def build(self) -> Model:
        """Buduje i kompiluje model LSTM.

        Kompilacja z optymalizatorem Adam i funkcją straty MSE:
            MSE = (1/N) * Σ (y_pred - y_true)²

        Adam (Adaptive Moment Estimation) automatycznie dobiera
        learning rate dla każdego parametru – standardowy wybór
        dla modeli rekurencyjnych (RNN/LSTM).

        Returns:
            Skompilowany model Keras gotowy do treningu.
        """
        # Warstwa wejściowa: każda próbka to macierz (lookback × n_features)
        inp = Input(
            shape=(self.lookback, self.n_features),
            name="input_sequence"
        )

        # ── Warstwa 1: LSTM z powrotem sekwencji ──────────────────────
        # return_sequences=True: zwraca stan ukryty h_t dla KAŻDEGO kroku t.
        # Jest to wymagane, gdy kolejna warstwa to też LSTM (musi widzieć
        # sekwencję, nie tylko ostatni stan).
        x = LSTM(
            self.lstm_units,
            return_sequences=True,
            name="lstm_1"
        )(inp)

        # Dropout po LSTM-1: losowo zeruje część wyjść przed LSTM-2
        x = Dropout(self.dropout_rate, name="dropout_1")(x)

        # ── Warstwa 2: LSTM bez powrotu sekwencji ─────────────────────
        # return_sequences=False (domyślnie): zwraca TYLKO ostatni stan
        # ukryty h_T, który koduje całą lookback-dniową sekwencję.
        x = LSTM(
            self.lstm_units,
            return_sequences=False,
            name="lstm_2"
        )(x)

        # Dropout po LSTM-2: podczas inferencji z training=True staje się
        # źródłem losowości dla Monte Carlo Dropout
        x = Dropout(self.dropout_rate, name="dropout_2")(x)

        # ── Warstwa wyjściowa: Dense ───────────────────────────────────
        # Aktywacja liniowa (brak aktywacji): ceny skalowane mogą być
        # dowolną nieujemną liczbą w [0, 1], nie wymagamy sigmoidalnego
        # ograniczenia (sieć sama uczy się właściwego zakresu).
        out = Dense(
            self.n_features,
            activation="linear",
            name="output"
        )(x)

        self.model = Model(inputs=inp, outputs=out, name="LSTM_MCDropout")

        # Kompilacja: optymalizator Adam z domyślnym lr=0.001
        self.model.compile(optimizer="adam", loss="mean_squared_error")

        return self.model

    def train(
        self,
        X_train:    np.ndarray,
        y_train:    np.ndarray,
        X_val:      np.ndarray,
        y_val:      np.ndarray,
        epochs:     int = 100,
        batch_size: int = 64,
    ) -> History:
        """Trenuje model LSTM z EarlyStopping na zbiorze walidacyjnym.

        EarlyStopping monitoruje val_loss i przerywa trening gdy model
        przestaje się poprawiać na danych walidacyjnych (ryzyko overfittingu).
        restore_best_weights=True przywraca wagi z najlepszej epoki.

        Parametr shuffle=False jest kluczowy dla danych szeregów czasowych:
        losowe tasowanie zniszczyłoby chronologiczny porządek próbek
        (model mógłby "widzieć przyszłość" podczas treningu).

        Args:
            X_train: Tensor treningowy, shape (n_train, lookback, n_features).
            y_train: Target treningowy, shape (n_train, n_features).
            X_val: Tensor walidacyjny, shape (n_val, lookback, n_features).
            y_val: Target walidacyjny, shape (n_val, n_features).
            epochs: Maksymalna liczba epok (EarlyStopping może przerwać wcześniej).
            batch_size: Liczba próbek przetwarzanych w jednym kroku gradientu.
                        Większy batch = bardziej stabilny gradient, mniej kroków.

        Returns:
            Obiekt History z historią wartości funkcji straty (trening / walidacja).

        Raises:
            RuntimeError: Jeśli model nie został zbudowany (wywołaj build() najpierw).
        """
        if self.model is None:
            raise RuntimeError("Model nie jest zbudowany. Wywołaj build() przed train().")

        # EarlyStopping: zatrzymaj trening gdy val_loss nie maleje przez
        # patience=12 epok; przywróć wagi z epoki o minimalnym val_loss
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),   # używamy zewnętrznego zbioru walidacyjnego
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            shuffle=False,    # WAŻNE: dane czasowe – zachowujemy chronologię
            verbose=1,
        )

        print(f"\n  Trening zakończony po {len(history.history['loss'])} epokach.")
        print(f"  Najlepsza val_loss: {min(history.history['val_loss']):.6f}")

        return history

    def mc_dropout_predict(
        self,
        X_input:   np.ndarray,
        n_samples: int,
        scaler:    MinMaxScaler,
    ) -> np.ndarray:
        """Wykonuje próbkowanie Monte Carlo Dropout i zwraca stopy zwrotu.

        Algorytm Monte Carlo Dropout:
        1. Dla każdej z n_samples iteracji:
           a. Wywołaj model z training=True → Dropout aktywny losowo
           b. Zapisz prognozowane ZNORMALIZOWANE ceny
        2. Cofnij normalizację skalera → prognozowane ceny w PLN/USD
        3. Oblicz log-stopy zwrotu: r = ln(P_pred / P_last)

        Rozkład uzyskanych stóp zwrotu (n_samples próbek) odzwierciedla
        niepewność modelu co do przyszłości:
          • Duże odchylenie std → model "niepewny" → wyższa kowariancja Σ
          • Małe odchylenie std → model "pewny" → niższa kowariancja Σ

        Args:
            X_input: Okno wejściowe, shape (1, lookback, n_features).
                     To ostatnie lookback dni z okresu treningowego.
            n_samples: Liczba próbek MC (rekomendowane: 100–500).
            scaler: MinMaxScaler dopasowany na danych treningowych
                    (potrzebny do odwrócenia normalizacji).

        Returns:
            ndarray kształtu (n_samples, n_features) z prognozowanymi
            dziennymi log-stopami zwrotu dla każdej spółki.

        Raises:
            RuntimeError: Jeśli model nie jest zbudowany.
        """
        if self.model is None:
            raise RuntimeError("Model nie jest zbudowany. Wywołaj build() i train().")

        preds_scaled: list[np.ndarray] = []

        for _ in range(n_samples):
            # ─────────────────────────────────────────────────────────
            # KLUCZOWA FLAGA: training=True
            # Informuje warstwy Dropout, że mają LOSOWO zerować neurony
            # nawet podczas inferencji (normalnie są wyłączone).
            # Każde wywołanie daje inny wynik → rozkład probabilistyczny.
            # ─────────────────────────────────────────────────────────
            pred: tf.Tensor = self.model(X_input, training=True)

            # pred.numpy()[0]: konwertuj Tensor → ndarray, weź pierwszą próbkę
            preds_scaled.append(pred.numpy()[0])  # shape: (n_features,)

        # Stos MC próbek: (n_samples, n_features) – znormalizowane ceny
        preds_scaled_arr: np.ndarray = np.array(preds_scaled)

        # Cofnij normalizację MinMaxScaler: znormalizowane → ceny w walucie
        preds_prices: np.ndarray = scaler.inverse_transform(preds_scaled_arr)

        # Ostatnia obserwacja treningowa (punkt bazowy dla stóp zwrotu)
        # X_input[0, -1, :] = ostatni dzień okna wejściowego (znorm.)
        last_scaled: np.ndarray = X_input[0, -1:, :]          # shape: (1, n_features)
        last_price:  np.ndarray = scaler.inverse_transform(last_scaled)[0]  # ceny bazowe

        # Log-stopa zwrotu: r = ln(P_pred / P_last)
        # Broadcasting: last_price[np.newaxis, :] → (1, n_features)
        mc_returns: np.ndarray = np.log(
            preds_prices / last_price[np.newaxis, :]
        )  # shape: (n_samples, n_features)

        return mc_returns

    def compute_mc_statistics(
        self,
        mc_returns:    np.ndarray,
        hist_returns:  "pd.DataFrame | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Oblicza μ_LSTM i hybrydową Σ dla optymalizacji Markowitza.

        ════════════════════════════════════════════════════════════════
        KLUCZOWA DECYZJA PROJEKTOWA: Hybrydowa macierz kowariancji Σ
        ════════════════════════════════════════════════════════════════

        Problem z czystą Σ_MC (macierzą próbek Monte Carlo):
        ─────────────────────────────────────────────────────
        MC Dropout generuje S=100 prognoz w jednym kroku czasowym.
        Każda próbka to wektor [r_1, r_2, ..., r_N] prognozowanych
        stóp zwrotu dla N spółek.

        Kowariancja tych S próbek mierzy, jak bardzo Dropout
        "waha się" co do prognoz POSZCZEGÓLNYCH spółek i jak
        te wahania są ze sobą POWIĄZANE W RAMACH MODELU.

        To NIE to samo co historyczna współzależność kursów:
          • Σ_MC  ← "Jak bardzo model jest niepewny co do pary (A, B)?"
          • Σ_hist ← "Jak bardzo kursy A i B historycznie poruszają się razem?"

        Dla DJIA z 30 spółkami korelacje rynkowe (Σ_hist) są dobrze
        oszacowane na 11 latach danych (~2 772 sesji). Σ_MC bazuje
        na zaledwie 100 próbkach z jednego punktu czasowego – jest
        statystycznie nieodporna i nie odzwierciedla prawdziwej
        współzależności rynkowej.

        Rozwiązanie: Hybrydowa macierz Σ_hybrid
        ─────────────────────────────────────────
        Łączymy najlepsze cechy obu źródeł:

          Σ_hybrid[i,j] = corr_hist[i,j] * σ_MC[i] * σ_MC[j]

        gdzie:
          corr_hist[i,j] = historyczny współczynnik korelacji Pearsona
                           między spółkami i oraz j (z ~2 772 sesji)
          σ_MC[i]        = odchylenie standardowe MC Dropout dla spółki i
                           (miara niepewności modelu co do spółki i)

        Interpretacja:
          • STRUKTURA KORELACJI  ← z historii (wiarygodna, duże N)
          • SKALA RYZYKA         ← z MC Dropout (subiektywna pewność modelu)

        Innymi słowy: LSTM mówi "jak bardzo jestem niepewny co do każdej
        spółki", ale struktura zależności między spółkami pochodzi
        z wieloletnich danych rynkowych – tak jak w realnym zarządzaniu portfelem.

        Jeśli hist_returns=None (brak danych historycznych), używamy
        czystej Σ_MC jako awaryjnego rozwiązania z ostrzeżeniem.

        Args:
            mc_returns: ndarray (n_samples, n_assets) – próbki MC Dropout.
                        Każdy wiersz to jeden przebieg inferencji z Dropout.
            hist_returns: DataFrame log-stóp zwrotu z okresu treningowego.
                          Kolumny: nazwy spółek. Shape: (n_days, n_assets).
                          Jeśli None, używa czystej Σ_MC (nie zalecane).

        Returns:
            Krotka (mu_lstm, cov_hybrid):
                mu_lstm:    ndarray (n_assets,) – oczekiwane dzienne stopy
                            zwrotu jako średnia próbek MC Dropout.
                cov_hybrid: ndarray (n_assets, n_assets) – hybrydowa macierz
                            kowariancji z historyczną strukturą korelacji
                            i skalą ryzyka MC Dropout.
        """
        import pandas as pd  # lokalny import – unikamy cykli modułów

        # ── Krok 1: Oczekiwana stopa zwrotu z MC Dropout ──────────────
        # Średnia po S próbkach: μ_LSTM = (1/S) * Σ_s r^(s)
        mu_lstm: np.ndarray = mc_returns.mean(axis=0)  # shape: (n_assets,)

        # ── Krok 2: Odchylenia standardowe z MC Dropout ───────────────
        # σ_MC[i] = std próbek MC dla spółki i
        # Odzwierciedla niepewność modelu: duże σ_MC → model "waha się"
        sigma_mc: np.ndarray = mc_returns.std(axis=0, ddof=1)  # shape: (n_assets,)

        # ── Krok 3: Historyczna macierz korelacji ─────────────────────
        if hist_returns is not None:
            # Macierz korelacji Pearsona z danych historycznych
            # corr_hist[i,j] = Cov(r_i, r_j) / (σ_i * σ_j)  ∈ [-1, 1]
            # Szacowana na ~2 772 sesjach (11 lat) – statystycznie wiarygodna
            corr_hist: np.ndarray = hist_returns.corr().values  # (n_assets, n_assets)

            # ── Krok 4: Złożenie Σ_hybrid ────────────────────────────
            # Wzór: Σ[i,j] = corr[i,j] * σ_i * σ_j
            # Macierz zewnętrzna sigma_mc @ sigma_mc.T daje σ_i * σ_j
            # Element-wise mnożenie przez corr_hist daje pełną kowariancję
            sigma_outer: np.ndarray = np.outer(sigma_mc, sigma_mc)  # σ_i * σ_j
            cov_hybrid: np.ndarray  = corr_hist * sigma_outer        # Σ[i,j]

            print(
                f"\n  [Σ] Hybrydowa macierz kowariancji:"
                f"\n      • Korelacje: historyczne ({hist_returns.shape[0]} sesji)"
                f"\n      • Wariancje:  MC Dropout ({len(mc_returns)} próbek)"
                f"\n      • Średnia σ_MC (dzienna): {sigma_mc.mean():.6f}"
                f"  ({sigma_mc.mean() * np.sqrt(252):.2%} rocznie)"
            )

        else:
            # Awaryjnie: czysta Σ_MC gdy brak danych historycznych
            # UWAGA: słaba estymacja korelacji przy małym N próbek!
            print(
                "\n  [WARN] Brak hist_returns – używam czystej Σ_MC."
                "\n         Korelacje między spółkami mogą być błędnie oszacowane."
            )
            cov_hybrid = np.cov(mc_returns, rowvar=False)

        return mu_lstm, cov_hybrid