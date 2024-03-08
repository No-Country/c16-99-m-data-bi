from django.urls import path
from . import views

urlpatterns = [
    path('predictions_nb', views.predictions_nb, name='predictions_nb'),
    path('predictions_gboost', views.predictions_gboost, name='predictions_gboost'),
    path('predictions_svm', views.predictions_svm, name='predictions_svm'),
    path('predictions_gboost_download/', views.predictions_gboost_download, name='predictions_gboost_download'),
    path('predictions_nb_download/', views.predictions_nb_download, name='predictions_nb_download'),
    path('predictions_svm_download/', views.predictions_svm_download, name='predictions_svm_download'),
]
