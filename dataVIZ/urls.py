from django.conf.urls import *
from dataVIZ import views
from django.contrib.staticfiles.urls import static
from django.conf import settings
from django.conf.urls.static import static

app_name = "dataVIZ"

urlpatterns = [
    url(r'^$',views.index,name='index'),
    #url(r'insert/',views.insert,name='insert'),
    url(r'records/$',views.records,name='records'),
    #url(r'^listview/',views.records,name='listview'),
    #url(r'records/',views.IndexView.as_view(),name='records'),
    
    # /records/id/
    #url(r'^(?P<student_id>[0-9]+)/$',views.details,name='studentDetails'),
    
    #studentDetails/student_id
    url(r'^studentDetails/([0-9]+)/$',views.student_details_page,name='studentDetails'),
    
    #VIEWS (ListView, DetailView,CreateView etc.)
    url(r'^listview/',views.IndexView.as_view(),name='listview'),
    #url(r'^detailview/(?P<pk>[0-9]+)/$',views.DetailView.as_view(),name='detailview'),
    url(r'^add/',views.StudentInformationCreate.as_view(),name='addstudent'),
    url(r'^update/(?P<pk>[0-9]+)/$',views.StudentUdpate.as_view(),name='update-student'),
    url(r'^delete/(?P<pk>[0-9]+)/$',views.DeleteStudent.as_view(),name='delete-student'),
    url(r'^predict/([0-9]+)/$',views.predict,name='predict')
]


urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)