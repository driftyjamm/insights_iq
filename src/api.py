import requests


def get_crypto_price():
    try:
        url = "https://api.coindesk.com/v1/bpi/currentprice.json"

        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return float(data['bpi']['USD']['rate'].replace(',', ''))

        return None

    except Exception:
        return None