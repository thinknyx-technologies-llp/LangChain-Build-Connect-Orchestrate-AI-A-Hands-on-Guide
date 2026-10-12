import json
import requests

from  fastmcp import FastMCP

mcp = FastMCP("Product Search Server")

@mcp.tool
def search_products(query: str) -> str:
    """
    Seach for products using a product name, category, or general product-related keywords.

    Examples: 
    - laptop
    - headphones
    - smartphone
    - coffee maker

    Extact the relevant product or category keyword from the user's natural language request
    """

    response = requests.get(
        "https://dummyjson.com/products/search",
        params = {
            "q" :query
        },
        timeout = 10
    )

    response.raise_for_status()

    data = response.json()

    products = data.get("products", [])

    results = [
        {
            "id" :  product["id"],
            "name": product["title"],
            "price": product["price"],
            "rating": product["rating"],
            "stock": product["stock"],
            "category": product["category"]
        }
        for product in products[:5]
    ]

    return json.dumps(results, indent = 2)

if __name__ =="__main__":
    mcp.run()

