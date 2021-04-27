from django.contrib import admin
from products_database.models import Product
from products_database.models import CheckedProduct


"""
	Models registred

	Product
	CheckedProduct

	Permission 
	-------
	To be stocked in the db

"""

admin.site.register(Product)
admin.site.register(CheckedProduct)