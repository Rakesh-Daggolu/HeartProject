from django.contrib import admin
from HeartApp.models import *
from django.shortcuts import render
from django.urls import path
# Register your models here.
class HeartAdmin(admin.ModelAdmin):
    list_display=['age','sex']

admin.site.register(Heart,HeartAdmin)
