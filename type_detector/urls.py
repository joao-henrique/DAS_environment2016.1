from django.conf.urls import url

from . import views

app_name = 'type_detector'
urlpatterns = [
    url(r'^detect$', views.detect, name="detect"),
]
