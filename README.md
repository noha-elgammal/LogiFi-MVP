# LogiFi – Supply Chain Risk Management

LogiFi is a proof‑of‑concept web application that demonstrates how modern tools can make supply‑chain risk analysis more transparent and actionable.  It was built for a hackathon and uses [Streamlit](https://streamlit.io) to create a bilingual (English/Arabic) interface.  The app walks a user through the “before → after” journey of a typical organisation: from manual risk tracking and delayed insight to instant risk estimation, scenario simulation and targeted recommendations.

## Project structure

The repository contains the following key files:

| File | Purpose |
| --- | --- |
| `new_site.py` | The main Streamlit application.  It defines all user interface pages, handles user sessions, calculates or predicts risk scores, draws charts, generates PDF reports and serves a simple Q&A assistant. |
| `risk_model.pkl` | A pickled RandomForest model used to estimate risk scores.  If the model cannot be loaded or prediction fails, the app falls back to a heuristic calculation. |
| `login_bg.png`, `logo.png` | Images used for the login page background and the LogiFi logo. |
| `requirements.txt` | List of Python packages needed to run the app locally. |
| `README.md` | This file.  It explains the project’s goals, how it works and how to run it. |

You may add a `.gitignore` file to exclude temporary files such as Python bytecode (`__pycache__/`), virtual environment folders (`venv/`) or local database snapshots.  Those files are not included here because they are created on‑the‑fly when you run the app.

## Running the app locally

This prototype is designed to run on your own machine.  There is **no requirement** to deploy it online for hackathon evaluation.  To run the application:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your‑organisation/logifi.git
   cd logifi
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   The requirements file lists Streamlit, Pandas, NetworkX, Plotly, FPDF, Matplotlib, scikit‑learn and other libraries used in this project.

4. **Run the application:**

   ```bash
   streamlit run new_site.py
   ```

5. **Open the app:**

   After running the above command, Streamlit will print a local URL (e.g. `http://localhost:8501`) in the terminal.  Open that link in your browser to view the app.  All state is stored in your session, so data will be lost when you stop the server.

## How it works

### Bilingual user interface

The sidebar lets users choose between **English** and **العربية**.  All labels, help messages, button texts and suggestions are pulled from a translation dictionary defined in `new_site.py`.  This makes it easy to add or modify translations.

### Login and user modes

Upon launching the app, you are greeted with a login page.  The username and password fields are checked against an in‑memory SQLite database created at runtime.  The app supports two account types – **Business** and **User** – which influence which pages appear in the sidebar.  For the purposes of the hackathon, account data are not persisted across sessions; they exist only in memory.

### Data upload

Users can upload a CSV file containing their supplier data.  The expected columns include:

* `supplier_name`: a string identifying the supplier;
* `inventory_value`: numeric inventory value;
* `monthly_revenue`: numeric monthly revenue of the buyer organisation;
* `num_suppliers`: integer count of suppliers;
* `delay_days`: integer number of delay days;
* `fx_exposure`: percentage exposure to currency fluctuations;
* `commodity_dependence`: percentage dependence on commodity prices.

After uploading, the data are shown in an editable table in the sidebar.  Users can correct values before running the risk analysis.

### Risk analysis

When the user proceeds to the **Risk Analysis** page, the app calculates a `predicted_risk` for each supplier.  It tries to load the RandomForest model contained in `risk_model.pkl` and call its `predict` method with the six features above.  If the model file is missing or prediction fails, a heuristic formula is used instead: risk increases with inventory value and delay days, adds proportional cost for FX and commodity exposure, and decreases slightly with more suppliers.

The results are presented in two ways:

* **Bar chart:** a standard bar chart shows the predicted risk for each supplier.
* **Network graph:** the ten suppliers with the highest average risk are arranged in a directed graph.  Node sizes and colours scale with risk values, giving a quick visual indication of which supplier drives the most risk.

The user can also simulate **What‑If scenarios**.  Three predefined problems – Port Delay, Currency Fluctuation, and Commodity Price Spike – allow the user to specify additional delay days, FX exposure or commodity dependence.  The app estimates the additional cost (in Saudi Riyals) if the specified increase were to occur.

### Chat assistant

On the **Chat Assistant** page, users can type natural‑language questions about their data.  The assistant supports two modes:

* **Beginner:** returns friendly explanations and glossary definitions for terms such as “risk”, “FX exposure” or “delay days”.
* **Pro:** offers concise answers for experienced users.

The assistant recognises patterns such as “top 3 risky suppliers”, “average risk”, “suppliers above 50k risk” or “suppliers with delay days over 10”.  It also returns suggestions to help users formulate further questions.  The assistant is rule‑based and does not use external APIs.

### PDF reporting

A “Generate Report” button on the Risk Analysis page produces a one‑page PDF summarising the analysis.  The report includes a timestamp, aggregate statistics and a bar chart.  It is generated with [FPDF](https://pyfpdf.github.io) and saved to the user’s computer via Streamlit’s `download_button`.

## FAQ

**Do I need to host the application?**  No.  The hackathon request is to upload your code to GitHub with a clear explanation.  Running the app locally via Streamlit is sufficient for evaluators.  If you wish to share a live demo, you could deploy to a service such as Streamlit Community Cloud or Heroku, but this is optional.

**Where does user data go?**  The prototype uses an in‑memory SQLite database for login data and Streamlit’s session state for uploaded files.  There is no external database or cloud storage.  All data vanish when you stop the server.

**Can I extend the model?**  Yes.  `risk_model.pkl` can be replaced with your own model as long as it implements a `predict` method that accepts a list of six numeric features in the order described above.  If no model is found, the heuristic will be used.

## License

This project is shared for hackathon evaluation and learning.  Feel free to fork it and experiment, but please do not use it in production without permission.