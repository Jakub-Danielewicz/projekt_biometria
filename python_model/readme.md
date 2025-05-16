# Instrukcja uruchomienia projektu

Projekt korzysta ze środowiska **Conda** oraz pliku `environment.yml` do zarządzania zależnościami.

---

## ✅ Wymagania wstępne

Aby uruchomić projekt, musisz mieć zainstalowaną **Condę** (Miniconda lub Anaconda).

### 🔧 Instalacja Condy

#### Opcja 1: Miniconda (zalecana – lżejsza)

1. Wejdź na stronę: https://docs.conda.io/en/latest/miniconda.html  
2. Pobierz instalator odpowiedni dla Twojego systemu operacyjnego.  
3. Zainstaluj, postępując zgodnie z instrukcjami.

#### Opcja 2: Anaconda

1. Wejdź na stronę: https://www.anaconda.com/products/distribution  
2. Pobierz i zainstaluj Anacondę.

Po instalacji sprawdź, czy Conda działa, wpisując w terminalu:

```bash
conda --version
```

---

## ⚙️ Utworzenie środowiska Conda

Po pobraniu (sklonowaniu) repozytorium:

```bash
git clone <adres_repozytorium>
cd <nazwa_katalogu_projektu>
```

Następnie utwórz środowisko na podstawie pliku `environment.yml`:

```bash
conda env create -f environment.yml
```

Po zakończeniu instalacji aktywuj środowisko:

```bash
conda activate <nazwa_środowiska>
```

ℹ️ **Uwaga:** Nazwa środowiska znajduje się w pliku `environment.yml` w linii `name:`.

---

## 🚀 Uruchamianie projektu

Po aktywacji środowiska możesz uruchomić projekt, np.:

```bash
python main.py
```

lub inny plik startowy zgodny z projektem.

---

## 🔁 Przydatne komendy Conda

- Wyświetlenie dostępnych środowisk:
  ```bash
  conda env list
  ```

- Usunięcie środowiska:
  ```bash
  conda remove --name <nazwa_środowiska> --all
  ```

---

## 📩 Kontakt

W razie problemów z konfiguracją – napisz do mnie
