

# Raport z projektu: Rozpoznawanie znaków pisma odręcznego

## Opis projektu

Celem projektu było stworzenie systemu rozpoznawania znaków pisma odręcznego z wykorzystaniem sieci neuronowych. Projekt obejmował zarówno przygotowanie własnego zbioru danych, jak i eksperymenty z dużym, ogólnodostępnym zbiorem z Kaggle. Przeprowadzono porównanie efektywności modeli na obu zbiorach oraz analizę uzyskanych wyników.

---

## Zbieranie własnego zbioru danych

Jednym z kluczowych etapów projektu było zebranie i przygotowanie własnego zbioru danych. W tym celu:
- **Stworzono specjalne szablony** do ręcznego wypełniania znakami przez różnych użytkowników.
- **Zeskanowano i przetworzono arkusze** – zaimplementowano narzędzia do automatycznego wykrywania i wycinania pojedynczych znaków z obrazów.
- **Przeprowadzono ręczną weryfikację i czyszczenie** – usunięto błędne lub nieczytelne próbki, a także zadbano o równomierny rozkład klas.
- Ostateczny zbiór zawierał **[tu wpisz liczbę] znaków** w **[tu wpisz liczbę] klasach**.

---

## Eksperymenty z dużym zbiorem z Kaggle

Dla porównania efektywności modeli, wykorzystano również duży, publicznie dostępny zbiór danych z platformy Kaggle:
- Wybrano zbiór **[tu wpisz nazwę zbioru, np. "A-Z Handwritten Data"]**.
- Zbiór ten charakteryzuje się dużą liczbą próbek oraz większą różnorodnością stylów pisma.
- Przeprowadzono analogiczne eksperymenty treningowe i walidacyjne jak dla własnego zbioru.

---

## Porównanie efektywności

| Zbiór danych         | Dokładność (accuracy) | Uwagi                                      |
|----------------------|----------------------|---------------------------------------------|
| Własny zbiór         | 72%    | Mniej danych, większa jednorodność stylów   |
| Kaggle               | 91%    | Większa różnorodność, lepsza generalizacja  |

- Modele trenowane na dużym zbiorze z Kaggle osiągały wyższą dokładność oraz lepiej radziły sobie z nieznanymi przykładami.
- Własny zbiór pozwolił jednak na lepsze dopasowanie do specyficznych stylów pisma obecnych w danych testowych.

---

## Wnioski i obserwacje

- **Jakość i różnorodność danych** mają kluczowe znaczenie dla skuteczności modeli rozpoznających pismo odręczne.
- Zbieranie własnego zbioru pozwala lepiej zrozumieć wyzwania związane z przygotowaniem danych oraz ich etykietowaniem.
- Modele trenowane na dużych, zróżnicowanych zbiorach generalizują lepiej, ale mogą mieć trudności z bardzo specyficznymi stylami pisma.
- Warto rozważyć **łączenie własnych danych z danymi publicznymi** w celu uzyskania najlepszego efektu.

---

## Podsumowanie

Projekt pozwolił na praktyczne poznanie całego procesu budowy systemu OCR dla pisma odręcznego: od zbierania i przetwarzania danych, przez projektowanie modeli, aż po analizę wyników i wyciąganie wniosków. Praca z własnym zbiorem danych była szczególnie cenna i pozwoliła na głębsze zrozumienie problemu oraz wyzwań związanych z uczeniem maszynowym
