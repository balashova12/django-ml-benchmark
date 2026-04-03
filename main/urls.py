from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("run/<str:algorithm>/", views.run_algorithm, name="run_algorithm"),
]