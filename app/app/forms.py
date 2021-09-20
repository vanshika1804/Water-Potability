from django import forms
import datetime
from django.forms.fields import ChoiceField, IntegerField
from django.utils.text import slugify
import datetime


class InputDataForm(forms.Form):
    

    ph = forms.FloatField()
    hardness = forms.FloatField()
    solids = forms.FloatField()
    chloramines = forms.FloatField()
    sulphate = forms.FloatField()
    conductivity = forms.FloatField()
    organic_carbon = forms.FloatField()
    trihalomethanes = forms.FloatField()
    turbidity = forms.FloatField()
    


