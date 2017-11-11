from django import forms
from dataVIZ.models import studentInformation

class studentInformationForm(forms.ModelForm):
    firstname = forms.CharField(max_length=20)
    lastname = forms.CharField(max_length=20)
    exam1_marks = forms.IntegerField(max_value=100,min_value=0)
    exam2_marks = forms.IntegerField(max_value=100,min_value=0)