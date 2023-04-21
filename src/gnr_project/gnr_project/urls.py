from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("segmentation_app/", include("segmentation_app.urls")),
    path("admin/", admin.site.urls),
]