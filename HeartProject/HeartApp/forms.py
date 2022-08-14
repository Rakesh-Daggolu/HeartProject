from django import forms
from HeartApp.models import *

class HeartForm(forms.ModelForm):
    class Meta:
        model=Heart
        exclude=['target']
