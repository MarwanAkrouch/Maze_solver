from django import forms
from products_database.models import Product


class ProductForm(forms.ModelForm):

    """
    class ProductForm

    Attributs
    -------
    Products

	form
    -------
	Select multiple choices

    """

    Products		= forms.ModelMultipleChoiceField(required=True,
    										queryset=Product.objects.values_list('name', flat=True).order_by('id'))
    
    
    class Meta :
    	
    	model  =  Product
    	fields =  []

