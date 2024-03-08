from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('product', views.product, name='product'),
    path('service', views.service, name='service'),
    path('price', views.price, name='price'),
    path('business', views.business, name='business'),
    path('contact', views.contact, name='contact'),
    path('register', views.register, name='register'),
    path('login', views.login, name='login'),
]