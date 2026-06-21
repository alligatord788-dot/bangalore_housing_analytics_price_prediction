# Bangalore Housing Analytics & Price Prediction Website Link

## Local Link On This Laptop

When the simple prediction website server is running, open:

```text
http://127.0.0.1:8000
```

Start the website with:

```bash
cd E:\Internship\projects\bangalore_housing_analytics_price_prediction
python web_app.py
```

## Streamlit Analytics Dashboard Link

When the analytics dashboard is running, open:

```text
http://localhost:8501
```

Start the dashboard with:

```bash
cd E:\Internship\projects\bangalore_housing_analytics_price_prediction
python -m streamlit run dashboard_app.py
```

## Can This Link Work On Another Device?

The `127.0.0.1` link works only on the same laptop where the server is running.

Reason:

`127.0.0.1` means "this same machine". If you open it on your phone or another laptop, it will look for a server running on that other device, not on this laptop.

## Using It On Another Device In The Same Wi-Fi

To use the website from another device connected to the same Wi-Fi:

1. Start the server on this laptop:

```bash
cd E:\Internship\projects\bangalore_housing_analytics_price_prediction
python web_app.py
```

2. Find this laptop's IPv4 address:

```bash
ipconfig
```

3. Look for something like:

```text
IPv4 Address . . . . . . . . . . . : 192.168.1.5
```

4. On the other device, open:

```text
http://192.168.1.5:8000
```

Replace `192.168.1.5` with your actual IPv4 address.

Important:

The current server code uses `127.0.0.1`, which is private to this laptop. To allow other devices on the same Wi-Fi, change this line in `web_app.py`:

```python
HOST = "127.0.0.1"
```

to:

```python
HOST = "0.0.0.0"
```

Then restart the server.

Windows Firewall may ask for permission. Allow it only for a trusted private network.

## Public Link For Anyone

For a public link that works anywhere, the app must be deployed online.

Simple options:

- Streamlit Community Cloud
- Render
- Hugging Face Spaces

For interview/demo preparation, the local link is enough unless you specifically want a public hosted version.
