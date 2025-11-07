# likviditetsprognose

# Projektbeskrivelse: Streamlit‑applikation til kommunal likviditetsprognose
## Formål

Målet er at udvikle en streamlit‑applikation, der henter offentligt tilgængelige data om danske kommuners økonomi, demografi og erhvervsudvikling, og derfra udregner en årlig likviditetsprognose for de næste tre år. Brugeren skal kunne filtrere på enkeltkommuner eller grupper af kommuner og sammenligne resultater på tværs.

## Datakilder

Applikationen baserer sig udelukkende på offentlige data, som hentes automatisk via API‑kald:

Statsbankens API (Danmarks Statistik) bruges til at hente kommunale regnskabs‑, budget‑ og demografidata. Relevant tabeller er bl.a.:

* REGK31 – kommunernes regnskabstal på funktionsniveau med årlig opløsning
api.statbank.dk
* REGK4 – balancekonti, som indeholder kommunernes likvide beholdninger (kassebeholdning) pr. år
api.statbank.dk
* BUDK1/BUDK32 – budgetter på hovedkonti/funktioner med årlig opløsning
api.statbank.dk
* FOLK1A – kvartalsvise befolkningstal efter kommune, alder og køn
api.statbank.dk
; bruges til at beregne likviditet pr. indbygger.

* DEMO19 – årlige tal for nye virksomheder og fuldtidsansatte pr. kommune
api.statbank.dk

* EJDSK1 og PSKAT for ejendomsskatteprovenu og skatteprocenter
api.statbank.dk
api.statbank.dk

Indenrigs‑ og Sundhedsministeriets meddelelser om tilskud og udligning offentliggøres årligt som PDF/Excel og indeholder bloktilskud og udligningsbeløb
ft.dk
* En manuel download bruges til at opdatere disse tal, da der ikke findes et API.

Alle andre beregninger bygger på disse offentlige kilder; ingen følsomme eller private data anvendes.

## Struktur og workflow
1. Dataintergration og ETL

Datascript (etl/download_data.py):

Et Python‑script som køres manuelt eller via GitHub‑Actions for at hente og opdatere alle datafiler.

Scriptet sender HTTP‑forespørgsler til Statsbankens API. For eksempel for REGK31:

import requests, pandas as pd
url = "https://api.statbank.dk/v1/data/REGK31/CSV?kommune=*&funktion=*&tid=2007-2024"
df = pd.read_csv(url, sep=';')


Data fra hver tabel gemmes i et standardformat (fx CSV eller Parquet) i mappen data/ i repositoryet.

Scriptet læser endvidere den årlige tilskud/udligning‑Excel fra Indenrigsministeriet (hentes manuelt eller med en URL) og transformerer beløbene til ét datasæt.

ETL‑processen renser data (ensretter kommunekoder, konverterer økonomital til kr.), beregner nøgletal (fx likviditet pr. indbygger) og gemmer et færdigt datasæt til brug i applikationen.

Automatisering via GitHub‑Actions (valgfrit):

En workflow‑fil .github/workflows/update_data.yml kan konfigureres til at køre download_data.py regelmæssigt (fx dagligt eller ugentligt).

Workflowen gemmer outputfiler som artefakter i repositoryet, så Streamlit‑applikationen kan hente dem uden først at kalde API’erne.

2. Prognosemodel

Baseline‑model: En simpel lineær regressionsmodel eller eksponentiel glidende gennemsnitsmodel bruges til at forudsige de næste tre års likviditet (kassebeholdning) pr. kommune. Modellen trænes på de seneste fem års årlige data fra REGK4 (likvid beholdning) og bruger evt. forklarende variable som:

Nettodriftsresultat og anlægsudgifter fra regnskab (REGK31)

Befolkningsvækst (FOLK1A)

Antal nye virksomheder (DEMO19)

Tilskud og udligning fra ministeriets tabel

Udvidelser: På sigt kan flere modeller tilføjes – fx ARIMA, Prophet eller maskinlæringsmodeller – og præsenteres som separate faner (tabs) i Streamlit‑grænsefladen. En abstraktion gør det let at tilføje flere modeller uden at ændre UI‑koden.

3. Streamlit‑applikation

Datatilgængelighed: Når brugeren åbner appen, læser den de forberedte datafiler fra data/. Hvis filerne ikke findes (første kørsel), kan appen alternativt udløse download_data.py for at hente dem, men dette medfører længere load‑tid. I produktionsbrug anbefales det at køre datascriptet på forhånd via GitHub‑Actions.

UI‑komponenter:

Sidepanel til filtrering: Brugeren kan vælge én eller flere kommuner, vælge tidsperiode og vælge hvilken model der skal vises.

Diagrammer: Tidssserier for kassebeholdning, nøgletal (likviditet pr. indbygger), skatteindtægter, budget vs. regnskab, samt prognosekurver for de næste tre år. Plotly kan bruges til interaktive grafer.

Tabelvisning: Mulighed for at se de underliggende data i tabelform og eksportere til CSV.

Fane‑navigation: Der oprettes en fane for hver prognosemodel, så brugeren kan sammenligne resultater.

4. Opsætning og kørsel

Repository: Koden ligger i et GitHub‑repository, der indeholder requirements.txt/pyproject.toml, download_data.py, Streamlit‑appen (app.py) og workflows.

Installation: Brugeren kloner repoet, installerer afhængigheder (Streamlit, pandas, scikit‑learn etc.) og kører først python download_data.py.

Afvikling: Start Streamlit‑appen med streamlit run app.py. Applikationen indlæser data, beregner prognosen og viser resultaterne. Da alle anvendte datasæt er offentlige, kan projektet hostes frit på GitHub uden licensproblemer.

Fremtidige udvidelser

Implementer flere prognosealgoritmer (ARIMA, Prophet) og vis dem som separate faner.

Inkluder machine‑learning baserede forklaringsvariable (fx socioøkonomiske indikatorer) for mere præcise prognoser.

Integrer datalagring i en database (SQLite eller PostgreSQL) for at forbedre performance og gøre det lettere at opdatere data.

Tilføj caching (fx st.cache_data) i Streamlit for at reducere load‑tiden ved gentagne åbninger.

Konklusion

Projektet kombinerer et ETL‑script til automatiseret dataindsamling fra Statsbankens API og ministeriets tilskudstabeller med en Streamlit‑frontend, der visualiserer kommunale kassebeholdninger og genererer treårsprognoser. Strukturen sikrer, at data kan hentes og forberedes en gang via en GitHub‑workflow, mens selve applikationen altid viser de nyeste tal til brugerne.
