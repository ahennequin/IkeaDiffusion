# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from collections import defaultdict
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
from scrapy.exporters import CsvItemExporter

EXPORTED_DATA_LOC = "scrapped_data.csv"


class IkeascraperPipeline:
    """This pipeline deletes duplicates products. Products are deemed duplicates if they both have the same 'url_id' and 'image_link'."""

    item_dict = defaultdict(list)

    def open_spider(self, spider):
        # Open output file and set exporter
        self.file = open(EXPORTED_DATA_LOC, "wb")
        self.exporter = CsvItemExporter(self.file)
        self.exporter.start_exporting()

    def close_spider(self, spider):
        # Release resources, close file nicely
        self.exporter.finish_exporting()
        self.file.close()

    def process_item(self, item, spider):
        # Wrap adapter around item, and retrieve needed information
        adapter = ItemAdapter(item)
        name = adapter.get("url_id")
        link = adapter.get("image_link")
        # If the 'product_name' appears in the dict
        if name in self.item_dict.keys():
            # Check if the associated link is listed
            if link in self.item_dict[name]:
                raise DropItem(f"Product has already been processed! ({name})")

        # If the item is not present, add it to the dict and export it
        self.item_dict[name].append(link)
        self.exporter.export_item(item)

        return item
