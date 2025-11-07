# Likviditetsprognose for danske kommuner

Dette repository indeholder en komplet demonstrationsløsning til at hente åbne
økonomiske data for danske kommuner, udarbejde et samlet datasæt og vise
likviditetsprognoser i en Streamlit-applikation.

## Struktur

- `etl/download_data.py` – script der henter og transformerer data fra
  Danmarks Statistik og det manuelt downloadede tilskudsregneark.
- `likviditetsprognose/` – Python-moduler med hjælpefunktioner til datahåndtering,
  prognosemodel og Plotly-visualiseringer.
- `app.py` – Streamlit-brugerfladen som læser det forberedte datasæt og viser
  grafer, nøgletal og prognoser.
- `.github/workflows/update_data.yml` – GitHub Actions workflow som kan køre
  ETL-processen kvartalsvist og gemme outputtet som artefakt.

## Kom godt i gang

1. **Installer afhængigheder**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Hent og forbered data**

   Download den seneste Excel-fil med tilskud og udligning fra
   [Indenrigs- og Sundhedsministeriet](https://im.dk) og gem den lokalt. Kør
   derefter ETL-scriptet:

   ```bash
   python etl/download_data.py --grants-file path/til/tilskud.xlsx
   ```

   Scriptet downloader data fra Statbankens API, gemmer rå CSV-filer i `data/raw`
   og bygger et samlet datasæt i `data/liquidity_dataset.parquet`.

3. **Start Streamlit-appen**

   ```bash
   streamlit run app.py
   ```

   Applikationen læser datasættet (eller bruger et indbygget eksempeldatasæt hvis
   filen ikke findes) og giver mulighed for at filtrere på kommuner, vælge
   prognosehorisont og eksportere resultater.

## Automatisering via GitHub Actions

Workflow-filen `.github/workflows/update_data.yml` kører ETL-scriptet hvert
kvartal (eller manuelt via *workflow dispatch*) og uploader det færdige datasæt
som et artefakt. Dette gør det muligt at holde Streamlit-appen opdateret uden at
belaste Statbankens API ved hver kørsel.

## Videre udvikling

- Tilføj mere avancerede prognosemodeller (fx ARIMA eller Prophet) og præsenter
  dem som ekstra faner i appen.
- Udvid datasættet med flere socioøkonomiske forklaringsvariable.
- Gem data i en database eller brug caching (`st.cache_data`) for at reducere
  svartid ved gentagne kald.

## Licens

Alle anvendte datakilder er offentligt tilgængelige, og koden i dette repository
kan frit anvendes til videre udvikling.
