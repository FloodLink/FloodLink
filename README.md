# FloodLink-API

Logic of Live Flood Risk Engine:


# 🌊 FloodLink Universal Flood-Risk Scaling Model

This document defines the global physical parameter ranges and scaling functions used by FloodLink’s dynamic flood-risk model.  
The goal is to ensure a universal, physics-consistent, and region-agnostic framework for evaluating flood potential anywhere on Earth.

---

## 🌍 Global Parameter Ranges (Physical Min–Max Reference)

These limits define the global operational envelope of FloodLink’s dynamic multipliers.  
They are based on climatological and hydrological data from NOAA, NASA SMAP, ECMWF ERA5, and GPCP datasets.

| Parameter | Symbol / Units | Global Minimum (Xₘᵢₙ) | Global Maximum (Xₘₐₓ) | Typical Extreme Context | Notes for Model Scaling |

|--------------------------------------------------------------------------------------------------------------------------------|

| Rainfall | R, mm per 6 h | `0` | `150` | 100–150 mm/6 h corresponds to intense tropical or monsoon downpours (e.g., Philippines, Bangladesh, Amazon Basin). | Main flood driver. Scaled 0.2 → 2.5× multiplier. |

| Soil Moisture (Volumetric Fraction) | θ, 0–1 | `0.00` | `0.60` | 0.55–0.60 observed in fully saturated lowland clays or marshes. | Defines infiltration/storage capacity. Scaled 0.8 → 1.4× multiplier. |

| Relative Humidity | RH, % | `30` | `100` | 30 % typical of dry desert air; 95–100 % during active storm systems. | Atmospheric persistence factor. Scaled 0.95 → 1.05× multiplier. |

---

## ⚙️ Linear Scaling Model

Each variable is transformed into a dimensionless multiplier using a linear interpolation between its global minimum and maximum.

M = M_min + ((X - X_min) / (X_max - X_min)) * (M_max - M_min)

- All values are clipped to remain within physical bounds:
  `X = max(X_min, min(X, X_max))`



## Flood disasters (2015–2025) — 24‑hour rainfall (mm)

Notes: standardized 24‑hour rainfall (mm) from an official met service, scientific paper, or widely cited technical summary. Ranges reflect the source’s 24‑hour reporting for specific gauges in the worst‑hit areas.

- **Valencia (Turís), Spain — 2024‑10‑29.** **771 mm / 24 h** (DANA); Spain daily record. Source: AEMET/MITECO — https://www.miteco.gob.es/es/prensa/ultimas-noticias/2024/noviembre/octubre-de-2024-fue-el-mas-lluvioso-de-la-serie-historica.html
  
- **Chennai, India — 2015‑12‑01.** **494 mm / 24 h** (city gauge). Sources: IMD — https://nwp.imd.gov.in/NWP-CHENNAI-RAINFALL-REPORT-2015.pdf ; AMS — https://journals.ametsoc.org/view/journals/mwre/146/4/mwr-d-16-0473.1.xml
  
- **Simbach am Inn, Germany — 2016‑06‑01.** **>180 mm / 24 h** locally. Source: DWD bulletin — https://rcccm.dwd.de/EN/ourservices/ravibulletinjahr/archiv/bulletin_2016.pdf
  
- **Mumbai (Santacruz), India — 2017‑08‑29/30.** **331.4 mm / 24 h** (08:30–08:30 IST). Source: IMD via ET — https://m.economictimes.com/news/politics-and-nation/at-331-4-mm-mumbai-gets-heaviest-rainfall-since-2005-deluge/articleshow/60293600.cms
  
- **Western Japan (Motoyama, Kōchi) — 2018‑07‑06/07.** **584 mm / 24 h**. Source: JMA summary — https://en.wikipedia.org/wiki/2018_Japan_floods
  
- **Aude (Trèbes), France — 2018‑10‑14/15.** **≈295 mm ~12 h** (daily similar). Source: Météo‑France — https://meteofrance.com/magazine/meteo-histoire/les-grands-evenements/14-15-octobre-2018-inondations-catastrophiques-sur-laude
  
- **Jakarta (Halim), Indonesia — 2020‑01‑01.** **377 mm / 24 h**. Sources: FloodList — https://floodlist.com/asia/indonesia-floods-jakarta-january-2020 ; AGU — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022GL101513
  
- **Kyushu, Japan — 2020‑07‑04…07.** **Up to 496 mm / 24 h**. Source: Ocean Dynamics — https://link.springer.com/article/10.1007/s10236-023-01551-1
  
- **W. Germany/Belgium (Köln‑Stammheim) — 2021‑07‑14/15.** **~154 mm / 24 h**. Source: overview — https://en.wikipedia.org/wiki/2021_European_floods
  
- **Henan (Zhengzhou), China — 2021‑07‑20.** **~617 mm / 24 h**; **201.9 mm in 1 h**. Sources: case study — https://www.researchgate.net/publication/375619257_Managing_Extreme_Rainfall_and_Flooding_Events_A_Case_Study_of_the_20_July_2021_Zhengzhou_Flood_in_China ; background — https://en.wikipedia.org/wiki/2021_Henan_floods
  
- **Seoul (Dongjak), South Korea — 2022‑08‑08/09.** **381.5 mm / 24 h**. Sources: KMA/press — https://www.koreaherald.com/article/2931287 ; https://apnews.com/article/3c4d984d7f8f964ad6d98d63bdd7fc74
  
- **Sindh (Padidan), Pakistan — 2022‑08‑19.** **355 mm / 24 h**. Source: PMD — https://cdpc.pmd.gov.pk/Pakistan_Monthly_Climate_Summary_August_2022.pdf
  
- **KwaZulu‑Natal, South Africa — 2022‑04‑11/12.** **>300 mm / 24 h** (several gauges). Sources: Atmosphere — https://www.mdpi.com/2073-4433/14/1/78 ; JBA — https://www.jbarisk.com/knowledge-hub/event-response/south-africa-kwazulu-natal-floods/
  
- **Auckland, New Zealand — 2023‑01‑27.** **~245–280 mm / 24 h**. Sources: NIWA — https://niwa.co.nz/news/auckland-suffers-wettest-month-history ; Review — https://tewaihanga.govt.nz/media/oeapj3qb/the-2023-auckland-anniversary-weekend-storm.pdf
  
- **Emilia‑Romagna, Italy — 2023‑05‑16/17.** **Up to 180–200+ mm / 24 h**. Source: CIMA — https://www.cimafoundation.org/en/news/the-italian-floods-of-may-2023-a-scientific-analysis/
  
- **Pelion (Zagora), Greece — 2023‑09‑05/06.** **~754 mm / 24 h**. Sources: press — https://greekcitytimes.com/2023/09/06/rain-record-in-greece-was-broken/ ; Guardian — https://www.theguardian.com/environment/2023/sep/08/weather-tracker-omega-block-brings-torrential-rain-to-greece-and-spain
  
- **Hong Kong SAR — 2023‑09‑07/08.** **638.5 mm / 24 h** (HKO HQ). Source: HKO — https://www.hko.gov.hk/en/wxinfo/pastwx/mws2023/mws202309.htm

- **New York City (JFK), USA — 2023‑09‑29.** **8.65 in ≈ 219 mm / 24 h**. Source: AP — https://apnews.com/article/a16ad4720eddd6c0f7ea3b761f414e80

- **Derna catchment, Libya — 2023‑09‑10/11.** **~350 mm / 24 h** (catchment‑avg). Source: Nature Communications — https://www.nature.com/articles/s41467-025-59261-9

- **United Arab Emirates — 2024‑04‑16.** **Up to ~259.5 mm / 24 h** (national record). Sources: ECMWF — https://www.ecmwf.int/en/newsletter/180/news/unprecedented-rainfall-united-arab-emirates ; AP — https://apnews.com/article/7bf3881efbea998dfa4c1ed8d538217c

- **Rio Grande do Sul, Brazil — 2024‑05‑02.** **249.4 mm / 24 h** (Soledade, INMET). Source: INMET — https://portal.inmet.gov.br/noticias/eventos-extremos-chuva-acima-da-m%C3%A9dia-marcam-maio-de-2024
