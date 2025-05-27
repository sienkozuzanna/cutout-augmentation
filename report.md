# Metody cutout w połączeniu z soft labelingiem - badanie skuteczności
## Wstęp
Celem projektu było przeprowadzenie eksperymentów mających na celu zweryfikowanie skuteczność augmentacji danych w treningu modeli. Badaną metodą augmentacji były różne sposoby cutoutu zdjęć, czyli wycinania losowych obszarów danego obrazu w celu wprowadzenia dodatkowej trudności dla modelu. Wycinanie polegało na zastąpieniu pewnej części pikseli pikselami czarnymi lub o losowych kolorach. Ta część badania inspirowana była [opublikowanymi artykułami](#bibliografia)  dotyczącymi różnych metod cutoutu i ich modyfikacjach. Z dołączonych papierów wynika, że taki sposób augmentacji danych poprawia naukę modeli, stąd motywacja na zweryfikowanie tej tezy dla różnorodnych sposobów cutoutu w naszym projekcie.

 Wyżej wspomniane metody połączono także z mechanizmem soft labeling zmieniającym etykiety klas proporcjonalnie do zakrytego pola. Inuticyjnie takie działanie powinno poprawić wyniki modelu na zbiorze walidacyjnym, ponieważ zostałby on lepiej nauczony rozpoznawania obrazów obraczonych szumem, zatem była to druga hipoteza, która została poddana ekspermentom w tym projekcie.

 ## Opis metod cutout
 ...
 ## Opis wykorzystanych danych
 W ramach projektu wykorzystano dwa zestawy danych obrazowaych: **CIFAR-10** oraz **Fashion-MNIST**. Oba zbiory należą do najczęściej stosowanych benchmarków w zadaniach klasyfikacji obrazów. 
 #### CIFAR-10
 Zbiór danych **CIFAR-10** zawiera **60 000 kolorowych obrazów RGB** o wymiarach **32×32 piksele**, podzielonych na **10 klas**:

- samolot  
- samochód  
- ptak  
- kot  
- jeleń  
- pies  
- żaba  
- koń  
- statek  
- ciężarówka

Podział zbioru:
- **50 000 obrazów treningowych**
- **10 000 obrazów testowych**

#### Fashion-MNIST

Zbiór danych **Fashion-MNIST** to nowoczesna alternatywa dla klasycznego MNIST. Zawiera **obrazy przedstawiające elementy odzieży** w odcieniach szarości o wymiarach **28×28 pikseli**, które są podzielone na **10 klas ubrań i akcesoriów**:
- T-shirt/top  
- Spodnie  
- Sweter  
- Sukienka  
- Płaszcz  
- Sandały  
- Koszula  
- Sneakersy  
- Torba  
- Botki
  
Podział zbioru:
- **60 000 obrazów treningowych**
- **10 000 obrazów testowych**


 
 ## Opis modeli
 ...
 ## Analiza wyników
 ...
 ## Dodatkowe eksperymenty i testy
 ... (te 999 i wgl opis problemów że to nie tak jak miało być )
 ## Podsumowanie
 (Kasia moze napisac na koniec)

 ## Bibliografia
 1. ["Colorful Cutout"](https://arxiv.org/html/2403.20012v1 )
 2. ["Benefits of Cutout and Cutmix"](https://arxiv.org/abs/2410.23672)
