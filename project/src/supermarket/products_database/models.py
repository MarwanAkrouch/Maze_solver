from django.db import models
from django.db.models import CharField, Model
from django_mysql.models import ListCharField


class Product(models.Model):

    """
    class Product

    Attributs
    -------
    order
    identity
    date
    name 

    class representative
    -------
	name

    """

    order		= models.DecimalField(max_digits=5, decimal_places=0)
    identity 	= models.DecimalField(max_digits=5, decimal_places=0)
    date 		= models.CharField(max_length=10)
    name 		= models.CharField(max_length=50)


    def __str__(self):
    	return self.name



class CheckedProduct(Model): 

    """
    class CheckedProduct

    Attributs
    -------
	productsList

    class representative
    -------
	productsList

    functions
    -------
	get_objects_list : list iterable

    """

    productsList = ListCharField( base_field=CharField(max_length=10), size=49, max_length=(49 * 11) )

    def get_objects_list(self):

    	productsList_iterable = []

    	for i in range (len(self.productsList)) :
    		productsList_iterable.append(self.productsList[i])

    	return productsList_iterable


    def __str__(self) :
    	return self.productsList


    class Meta :

    	get_latest_by = 'productsList'