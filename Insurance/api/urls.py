from django.urls import path
from . import views

urlpatterns = [
    path('predict/',views.predict,name='predict'),
    path('chatbot/',views.chatbot_response, name='chatbot_response'),
    ]
