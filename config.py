"""
config.py – Centralne parametry konfiguracyjne projektu
=========================================================
Wszystkie stałe używane w projekcie są zdefiniowane tutaj,
co ułatwia ich modyfikację bez konieczności przeszukiwania
wielu plików.
"""

# ─────────────────────────────────────────────────────────
# 1. SKŁADNIKI INDEKSU DOW JONES INDUSTRIAL AVERAGE (30 spółek)
# ─────────────────────────────────────────────────────────
# Ticker Yahoo Finance → nazwa spółki
# Dow Jones Industrial Average (DJIA) to cenowo ważony indeks
# 30 największych i najbardziej wpływowych spółek notowanych na
# giełdach NYSE i NASDAQ, obliczany od 1896 roku.

TICKERS: dict[str, str] = {
    "NVIDIA Corporation": "NVDA",          # półprzewodniki / AI
    "Apple Inc.": "AAPL",                  # technologia / konsumer
    "Microsoft Corporation": "MSFT",       # oprogramowanie / chmura
    "Amazon.com Inc.": "AMZN",             # e-commerce / chmura
    "Walmart Inc.": "WMT",                # handel detaliczny
    "JPMorgan Chase & Co.": "JPM",        # bankowość / finanse
    "Visa Inc.": "V",                      # płatności elektroniczne
    "Johnson & Johnson": "JNJ",           # farmaceutyki / MedTech
    "Cisco Systems Inc.": "CSCO",          # infrastruktura sieciowa
    "Caterpillar Inc.": "CAT",             # maszyny budowlane
    "Chevron Corporation": "CVX",         # ropa i gaz
    "UnitedHealth Group Inc.": "UNH",     # ubezpieczenia zdrowotne
    "The Coca-Cola Company": "KO",       # napoje / FMCG
    "Procter & Gamble Co.": "PG",        # produkty konsumenckie
    "The Home Depot Inc.": "HD",         # materiały budowlane
    "Goldman Sachs Group Inc.": "GS",    # bankowość inwestycyjna
    "Merck & Co. Inc.": "MRK",            # farmaceutyki
    "American Express Company": "AXP",    # karty kredytowe
    "IBM Corporation": "IBM",             # IT / consulting
    "McDonald's Corporation": "MCD",      # restauracje / franczyza
    "Verizon Communications Inc.": "VZ", # telekomunikacja
    "The Walt Disney Company": "DIS",     # media / rozrywka
    "Amgen Inc.": "AMGN",                  # biotechnologia
    "Boeing Company": "BA",              # lotnictwo / obronność
    "Salesforce Inc.": "CRM",             # CRM / chmura
    "Honeywell International Inc.": "HON",# konglomerat przemysłowy
    "3M Company": "MMM",                  # przemysł / innowacje
    "Sherwin-Williams Company": "SHW",    # farby / powłoki
    "Travelers Companies Inc.": "TRV",    # ubezpieczenia majątkowe
    "Nike Inc.": "NKE",                   # odzież sportowa
}

# Benchmark – indeks Dow Jones Industrial Average
# Yahoo Finance udostępnia go pod symbolem ^DJI
BENCHMARK_TICKER: str = "^DJI"

# ─────────────────────────────────────────────────────────
# 2. PODZIAŁ OKRESU BADANIA (15 LAT: 2011–2025)
# ─────────────────────────────────────────────────────────
# Schemat podziału danych na 3 rozłączne okresy:
#   TRENING   : 2011-01-01 → 2021-12-31  (11 lat, ~2 772 sesji)
#     – na tych danych trenujemy LSTM i kalibrujemy CAPM/Markowitz
#   WALIDACJA : 2022-01-01 → 2023-12-31  ( 2 lata, ~504 sesje)
#     – wczesne zatrzymanie treningu LSTM; wybór hiperparametrów
#   TEST      : 2024-01-01 → 2025-12-31  ( 2 lata, ~504 sesje)
#     – ocena ostateczna (out-of-sample), nienaruszona podczas budowy modelu

START_DATE:      str = "2011-01-01"
END_DATE:        str = "2025-12-31"

# TRAIN_START:     str = "2011-01-01"
# TRAIN_END:       str = "2021-12-31"
TRAIN_START:     str = "2018-01-01"
TRAIN_END:       str = "2022-12-31"

# VALIDATION_START: str = "2022-01-01"
# VALIDATION_END:   str = "2023-12-31"
VALIDATION_START: str = "2023-01-01"
VALIDATION_END:   str = "2023-12-31"

TEST_START:      str = "2024-01-01"
# TEST_START:      str = "2025-01-01"
TEST_END:        str = "2025-12-31"

# ─────────────────────────────────────────────────────────
# 3. HIPERPARAMETRY MODELU LSTM
# ─────────────────────────────────────────────────────────
# LOOKBACK: model "widzi" LOOKBACK ostatnich dni sesyjnych jako kontekst.
#   Wartość 60 ≈ 3 miesiące handlowe – wystarczy do uchwycenia
#   krótko- i średnioterminowych trendów cenowych.
# LOOKBACK: int = 60
LOOKBACK: int = 30

# LSTM_UNITS: liczba komórek pamięci w każdej warstwie LSTM.
#   Wyższa wartość = więcej parametrów = większa zdolność modelu,
#   ale też większe ryzyko przeuczenia i dłuższy czas treningu.
LSTM_UNITS: int = 50

# DROPOUT_RATE: odsetek neuronów zerowanych losowo w każdym kroku.
#   Pełni podwójną rolę:
#   a) regularyzacja podczas treningu (zmniejsza przeuczenie),
#   b) źródło losowości w Monte Carlo Dropout podczas inferencji
#      (generuje rozkład prognoz → estymacja niepewności).
DROPOUT_RATE: float = 0.05

# Parametry treningu sieci neuronowej
EPOCHS:     int = 100   # maksymalna liczba epok (EarlyStopping przerwie wcześniej)
BATCH_SIZE: int = 64    # liczba próbek na jeden krok gradientu

# ─────────────────────────────────────────────────────────
# 4. PARAMETRY MONTE CARLO DROPOUT
# ─────────────────────────────────────────────────────────
# MC_SAMPLES: liczba niezależnych przebiegów inferencji z aktywnym Dropout.
#   Każdy przebieg generuje nieco inną prognozę (bo Dropout losowo zeruje
#   neurony). Rozkład tych MC_SAMPLES prognoz aproksymuje bayesowski
#   posteriorny rozkład predykcji modelu.
#   100 próbek = dobry kompromis jakość/czas (błąd ≈ 1/√100 = 10 %).
MC_SAMPLES: int = 100

# ─────────────────────────────────────────────────────────
# 5. PARAMETRY FINANSOWE
# ─────────────────────────────────────────────────────────
# RISK_FREE_ANNUAL: roczna stopa wolna od ryzyka.
#   Przyjmujemy ≈ 4.5 % – przybliżona stopa T-Bill (3-miesięczny
#   bon skarbowy USA) uśredniona za lata 2011–2025.
#   Wzór na stopę dzienną: r_d = r_a / 252
#   (252 = konwencjonalna liczba sesji giełdowych w roku w USA)
RISK_FREE_ANNUAL: float = 0.045
RISK_FREE:        float = RISK_FREE_ANNUAL / 252   # dzienna stopa wolna od ryzyka

# Liczba sesji giełdowych w roku – standard rynków amerykańskich (NYSE/NASDAQ)
TRADING_DAYS: int = 252

# ─────────────────────────────────────────────────────────
# 6. PARAMETRY OPTYMALIZACJI PORTFELA
# ─────────────────────────────────────────────────────────
# N_RESTARTS: liczba losowych punktów startowych w optymalizacji Sharpe'a.
#   Optymalizacja Markowitza jest problemem nieliniowym z wieloma minimami
#   lokalnymi. Wielokrotne starty z różnych punktów zwiększają szansę
#   znalezienia optimum globalnego.
N_RESTARTS: int = 20

# ─────────────────────────────────────────────────────────
# 7. ZIARNO LOSOWOŚCI (REPRODUKOWALNOŚĆ EKSPERYMENTÓW)
# ─────────────────────────────────────────────────────────
# Ustawienie tego samego SEED we wszystkich bibliotekach gwarantuje,
# że uruchomienie kodu wielokrotnie da identyczne wyniki.
SEED: int = 42