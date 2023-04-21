from django.urls import path

from . import views

urlpatterns = [
    path("index/", views.segment_image, name="index"),
]
