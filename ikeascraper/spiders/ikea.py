from dataclasses import dataclass
import scrapy


def extract_image_link(sources: str, image_size: str):
    links = map(
        lambda l: l.split(" ")[0],  # Remove end of the string, after the actual link
        filter(
            lambda s: s.startswith(
                "https://"
            ),  # Filter on whether the string starts as an http address
            map(
                lambda s: s.strip(), sources.split("\n")
            ),  # Split items, then remove all leading/ending spaces
        ),
    )
    return next(filter(lambda l: l.endswith(image_size), links))


@dataclass
class IkeaProduct:
    url_id: str
    product_name: str
    description: str
    image_link: str


class IkeaWebsiteSpider(scrapy.Spider):
    name = "products"
    start_urls = [
        "https://www.ikea.com/fr/fr/p/skoenabaeck-canape-2-places-convertible-knisa-gris-fonce-70582542/",
        "https://www.ikea.com/fr/fr/p/vesteroey-matelas-a-ressorts-ensaches-mi-ferme-bleu-clair-00450615/",
    ]

    def parse(self, response, **kwargs):
        split_url = response.url.split("/")
        if "p" in split_url:  # Product page
            url_id = split_url[
                split_url.index("p") + 1
            ]  # Product name is directly after /p/ in the URL

            # Retrieve image on the page product
            element = response.xpath(
                "/html/body/main/div/div[1]/div/div[2]/div[1]/div/div[3]/div/div[1]/div/span/img"
            )
            yield IkeaProduct(
                url_id=url_id,
                product_name=element.attrib["alt"].split(" ")[
                    0
                ],  # Product name is the first word of the description
                description=element.attrib["alt"],
                image_link=extract_image_link(element.attrib["srcset"], "xl"),
            )
        else:
            self.logger.warning(f"Expected to find 'p' in URL!")
            raise NotImplementedError
