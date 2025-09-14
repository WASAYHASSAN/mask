from django.urls import path
from . import views

app_name = 'mask'

urlpatterns = [
    path('', views.upload_view, name='upload'),
]