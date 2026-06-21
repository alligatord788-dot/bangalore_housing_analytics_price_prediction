from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs
import html
import json
import pickle

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "model.pkl"
COLUMNS_PATH = PROJECT_DIR / "artifacts" / "columns.json"
HOST = "127.0.0.1"
PORT = 8000


def load_artifacts():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(COLUMNS_PATH, "r") as file:
        columns = json.load(file)["columns"]

    locations = [
        column for column in columns if column not in {"total_sqft", "bath", "bhk"}
    ]
    return model, columns, sorted(locations)


MODEL, COLUMNS, LOCATIONS = load_artifacts()


def predict_price(location, sqft, bath, bhk):
    input_data = {column: 0 for column in COLUMNS}
    input_data["total_sqft"] = sqft
    input_data["bath"] = bath
    input_data["bhk"] = bhk

    if location in input_data:
        input_data[location] = 1
    elif "other" in input_data:
        input_data["other"] = 1

    input_df = pd.DataFrame([input_data], columns=COLUMNS)
    return round(float(MODEL.predict(input_df)[0]), 2)


def render_page(result=None, error=None, form_data=None):
    form_data = form_data or {}
    selected_location = form_data.get("location", "1st Phase JP Nagar")

    location_options = []
    for location in LOCATIONS:
        selected = "selected" if location == selected_location else ""
        safe_location = html.escape(location)
        location_options.append(
            f'<option value="{safe_location}" {selected}>{safe_location}</option>'
        )

    result_html = ""
    if result is not None:
        result_html = f"""
        <section class="result">
          <span>Predicted price</span>
          <strong>{result} lakhs</strong>
        </section>
        """

    error_html = ""
    if error:
        error_html = f'<p class="error">{html.escape(error)}</p>'

    sqft = html.escape(str(form_data.get("sqft", "1000")))
    bath = html.escape(str(form_data.get("bath", "2")))
    bhk = html.escape(str(form_data.get("bhk", "2")))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bangalore House Price Predictor</title>
  <style>
    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f7fb;
      color: #182230;
    }}

    main {{
      max-width: 860px;
      margin: 0 auto;
      padding: 36px 18px;
    }}

    h1 {{
      margin: 0 0 8px;
      font-size: 2rem;
      letter-spacing: 0;
    }}

    .subtext {{
      margin: 0 0 24px;
      color: #526071;
      line-height: 1.5;
    }}

    form {{
      display: grid;
      gap: 16px;
      padding: 22px;
      background: #ffffff;
      border: 1px solid #d8e0ea;
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(24, 34, 48, 0.08);
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}

    label {{
      display: grid;
      gap: 7px;
      font-weight: 700;
      font-size: 0.95rem;
    }}

    input,
    select {{
      width: 100%;
      min-height: 42px;
      padding: 9px 11px;
      border: 1px solid #bac6d4;
      border-radius: 6px;
      font: inherit;
      background: #ffffff;
      color: #182230;
    }}

    button {{
      justify-self: start;
      min-height: 42px;
      padding: 0 18px;
      border: 0;
      border-radius: 6px;
      background: #166534;
      color: #ffffff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}

    button:hover {{
      background: #14532d;
    }}

    .result {{
      margin-top: 20px;
      display: grid;
      gap: 4px;
      padding: 18px;
      background: #eaf7ef;
      border: 1px solid #b7e1c2;
      border-radius: 8px;
    }}

    .result span {{
      color: #2f5d40;
      font-weight: 700;
    }}

    .result strong {{
      font-size: 2rem;
      letter-spacing: 0;
    }}

    .error {{
      margin: 0;
      color: #b42318;
      font-weight: 700;
    }}

    @media (max-width: 640px) {{
      main {{
        padding: 24px 14px;
      }}

      h1 {{
        font-size: 1.55rem;
      }}

      .grid {{
        grid-template-columns: 1fr;
      }}

      button {{
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Bangalore House Price Predictor</h1>
    <p class="subtext">Enter property details and get an estimated price from the trained regression model.</p>

    <form method="post">
      {error_html}
      <label>
        Location
        <select name="location">
          {"".join(location_options)}
        </select>
      </label>

      <div class="grid">
        <label>
          Total square feet
          <input name="sqft" type="number" min="100" step="1" value="{sqft}" required>
        </label>
        <label>
          Bathrooms
          <input name="bath" type="number" min="1" step="1" value="{bath}" required>
        </label>
        <label>
          BHK
          <input name="bhk" type="number" min="1" step="1" value="{bhk}" required>
        </label>
      </div>

      <button type="submit">Predict Price</button>
    </form>

    {result_html}
  </main>
</body>
</html>"""


class PricePredictionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_html(render_page())

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        form = {key: values[0] for key, values in parse_qs(body).items()}

        try:
            location = form["location"]
            sqft = float(form["sqft"])
            bath = int(form["bath"])
            bhk = int(form["bhk"])

            if sqft <= 0 or bath <= 0 or bhk <= 0:
                raise ValueError("Sqft, bathrooms, and BHK must be positive.")

            result = predict_price(location, sqft, bath, bhk)
            self.send_html(render_page(result=result, form_data=form))
        except (KeyError, ValueError) as exc:
            error = f"Please enter valid inputs. {exc}"
            self.send_html(render_page(error=error, form_data=form))

    def send_html(self, page):
        encoded_page = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded_page)))
        self.end_headers()
        self.wfile.write(encoded_page)

    def log_message(self, format, *args):
        return


def run_server():
    server = HTTPServer((HOST, PORT), PricePredictionHandler)
    print(f"Website running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()

