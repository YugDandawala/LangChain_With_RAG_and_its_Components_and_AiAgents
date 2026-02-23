from bs4 import BeautifulSoup
import requests
import pandas as pd

headers = {
    "User-Agent": "Mozilla/5.0"
}

base_url = "https://www.flipkart.com/search?q=phones&page="

data = {
    "P_name": [],
    "Color": [],
    "Storage": [],
    "Ram": [],
    "Ratings": [],
    "Display": [],
    "Battery": [],
    "Price": []
}

for page in range(1, 10):

    url = base_url + str(page)
    req = requests.get(url, headers=headers)
    soup = BeautifulSoup(req.text, "html.parser")

    # Get ALL blocks with data-id
    raw_blocks = soup.find_all("div", attrs={"data-id": True})

    products = [
        block for block in raw_blocks
        if block.find("div", class_="RG5Slk")
    ]

    for product in products:

        # ---------- NAME ----------
        name_block = product.find("div", class_="RG5Slk")
        if name_block:
            full_text = name_block.get_text(strip=True)
            if "(" in full_text:
                p_name = full_text.split("(")[0].strip()
                other = full_text.split("(")[1].replace(")", "")
                parts = [p.strip() for p in other.split(",")]
                color = parts[0] if len(parts) > 0 else "N/A"
                storage = parts[1] if len(parts) > 1 else "N/A"
            else:
                p_name = full_text
                color = storage = "N/A"
        else:
            p_name = color = storage = "N/A"

        Ram = display = battery = Rating = price = "N/A"

        for d in product.find_all("li", class_="DTBslk"):
            text = d.get_text(strip=True)
            if "GB RAM" in text:
                Ram = text.split(" ")[0]
            elif "cm" in text or "inch" in text:
                display = text.split(" ")[0]
            elif "mAh" in text:
                battery = text.split(" ")[0]

        for r in product.find_all("span", class_="PvbNMB"):
            txt = r.get_text(strip=True)
            if "Ratings" in txt:
                Rating = txt.split(" ")[0]

        for p in product.find_all("div", class_="hZ3P6w DeU9vF"):
            txt = p.get_text(strip=True)
            if "₹" in txt:
                price = txt.split("₹")[1].strip()

        data["P_name"].append(p_name)
        data["Color"].append(color)
        data["Storage"].append(storage)
        data["Ram"].append(Ram)
        data["Display"].append(display)
        data["Battery"].append(battery)
        data["Ratings"].append(Rating)
        data["Price"].append(price)

        # print(p_name, "|", color, "|", storage, "|", Ram, "|", display, "|", battery, "|", Rating, "|", price)

df=pd.DataFrame(data)
df.to_csv("Flipkart_Data.csv",index=True)