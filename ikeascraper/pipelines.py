# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from collections import defaultdict
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class IkeascraperPipeline:
    """This pipeline deletes duplicates products. Products are deemed duplicates if they both have the same name and image link."""

    item_dict = defaultdict(list)

    def process_item(self, item, spider):
        # Wrap adapter around item, and retrieve needed information
        adapter = ItemAdapter(item)
        name = adapter.get("product_name")
        link = adapter.get("image_link")
        # If the 'product_name' appears in the dict
        if name in self.item_dict.keys():
            # Check if the associated link is listed
            if link in self.item_dict[name]:
                raise DropItem(f"Product has already been processed! ({name})")

        # If the item is not present, add it
        self.item_dict[name].append(link)
        return item
