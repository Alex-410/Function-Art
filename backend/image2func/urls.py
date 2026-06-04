from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_and_process, name='upload_and_process'),
    path('upload_stream/', views.upload_and_process_stream, name='upload_and_process_stream'),
    path('system_info/', views.system_info, name='system_info'),
    path('get_frame/', views.get_frame, name='get_frame'),
    path('get_all_frames/', views.get_all_frames, name='get_all_frames'),
    path('clear_cache/', views.clear_cache, name='clear_cache'),
]
