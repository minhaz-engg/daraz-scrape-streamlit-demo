import json

from orchestra import scrape_product



def scrape_product_detail(url: str) -> str:
    # Put your product page URL here
    # url = "https://www.daraz.com.bd/products/high-quality-mens-shoes-imported-i378023244-s1895600858.html"

    result = scrape_product(url, render_if_missing=True, debug=True)

    js = json.dumps(result, ensure_ascii=False, indent=2)
    return js
    # print(js)
    # with open("product.json", "w", encoding="utf-8") as f:
    #     f.write(js)
    # print("Saved to product.json")

# url = "https://www.daraz.com.bd/products/understated-craftsmanship-and-trendy-new-indian-vichitra-silk-saree-high-quality-embroidery-work-on-blouse-progressively-better-i305660146-s1365577801.html"
# scrape_product_detail(url=url)