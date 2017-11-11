from django.http import Http404
from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
from dataVIZ.forms import studentInformationForm
from dataVIZ.models import studentInformation
from django.views import generic
from django.core.paginator import Paginator
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.core.urlresolvers import reverse_lazy
import os
#in order to get static files
from django.contrib.staticfiles.templatetags.staticfiles import static
from django.contrib.staticfiles import storage

from django.views.decorators import csrf

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy import optimize
#import pandas as pd
#from sklearn import linear_model
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
plt.style.use('seaborn-bright')


def predict(request,student_id):
    
    stud = studentInformation.objects.get(pk=student_id)
    #datafile = 'C:\Django Stuff\mysite\dataVIZ\ex2data1.txt'
    datafile = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'static/ex2data1.txt')
    #datafile = static('images/ex2data1.txt')
    data = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)

    X = np.transpose(np.array(data[:-1]))
    y = np.transpose(np.array(data[-1:]))

    #size of data
    m = y.size
    X = np.insert(X,0,1,axis=1)

    #cases where the student gets admitted
    pos = np.array([X[i] for i in range(X.shape[0]) if y[i]==1])

    #cases where the student does not get admitted
    neg = np.array([X[i] for i in range(X.shape[0]) if y[i]==0])

    def plot_data():
        
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(pos[:,1],pos[:,2],'ro',label='Admitted')
        ax.plot(neg[:,1],neg[:,2],'yo',label='Not Admitted')
        #ax.xlabel('Exam 1 Score')
        #ax.ylabel('Exam 2 Score')
        ax.set_xlabel('Exam 1 Score')
        ax.set_ylabel('Exam 2 Score')
        ax.grid('on')
        ax.plot(boundary_xs,boundary_ys,'m-',label='Decision Boundary')
        ax.plot(stud.exam1_marks,stud.exam2_marks,'bo',label= stud.firstname+' '+stud.lastname)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax.grid('on')
        #fig.savefig('C:\Django Stuff\heroku_deploy\mysite\static\images\imgdata.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
        fig.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'static/images/imgdata.png'),bbox_extra_artists=(lgd,), bbox_inches='tight')
    #
    def hypothesis(mytheta,myX):
        return expit(np.dot(myX,mytheta))

    def computeCost(mytheta,myX,myy,mylamda=0.):
        term1 = np.dot(-np.array(myy).T,np.log(hypothesis(mytheta,myX)))
        term2 = np.dot((1-np.array(myy)).T,np.log(1-hypothesis(mytheta,myX)))
        reg_term = (mylamda/2) * np.sum(np.dot(mytheta[1:].T,mytheta[1:]))
        return float((1./m) * (np.sum(term1 - term2 + reg_term)))

    initial_theta = np.zeros((X.shape[1],1))
    #print initial_theta

    computeCost(initial_theta,X,y)


    def optimizeTheta(mytheta,myX,myy,mylambda=0.):
        result = optimize.fmin(computeCost,x0=mytheta,args=(myX,myy,mylambda),maxiter=400,full_output=True)
        return result[0],result[1]


    theta, minCost = optimizeTheta(initial_theta,X,y)

    def predict(theta,X,threshold=0.5):
        p = expit(np.dot(X,theta.T)) >= threshold
        return p.astype('int')

    p = predict(theta,X)

    #Plotting the decision boundary
    boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])

    #slope of the decision line
    boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
    plot_data()

    img = mpimg.imread(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'static/images/imgdata.png'))
    context = {'student':stud}

    template = loader.get_template('dataVIZ/prediction.html')
    return HttpResponse(template.render(context,request))

#@csrf
def index(request):
    #return render(request,'dataVIZ/home.html')
    template = loader.get_template('dataVIZ/home.html')
    return HttpResponse(template.render(request))


#def insert(request):
#    return render(request,'dataVIZ/insert.html')

def insert(request):
    template = loader.get_template('dataVIZ/insert.html')
    return HttpResponse(template.render(request))


#def records(request):
    #return render(request,'dataVIZ/records.html',{'all_students':all_records})

def records(request):
    
    all_records = studentInformation.objects.all()
    template = loader.get_template('dataVIZ/records.html')
    context = {
        'all_students':all_records,
    }
    return HttpResponse(template.render(context,request))
    #return render(request,'dataVIZ/records.html',context)



def details(request, student_id):
    
    try:
        all_records = studentInformation.objects.get(pk=student_id)
        template = loader.get_template('dataVIZ/studentDetails.html')
        context = {'student_id':student_id}
        
    except studentInformation.DoesNotExist:
        raise Http404("Student Information Unavailable")
    
    return HttpResponse(template.render(context,request))
        
        #return HttpResponse("<h2>Details for Student id: "+str(student_id)+"</h2>")

        
#-----------------------------------------------------GENERIC VIEWS---------------------------------------------
    
class IndexView(generic.ListView):
    model = studentInformation
    template_name = 'dataVIZ/studentinformation_list.html'
    context_object_name = 'all_students'
    paginate_by = 5
    queryset = studentInformation.objects.all()
    
    def get_queryset(self):
        return studentInformation.objects.all()

    
#class DetailView(generic.DetailView):
#    model = studentInformation
#    template_name = 'dataVIZ/studentDetails.html'
    

class StudentInformationCreate(CreateView):
    model = studentInformation
    fields = ['firstname','lastname','exam1_marks','exam2_marks']
    
    success_url = reverse_lazy('dataVIZ:listview')
    
class StudentUdpate(UpdateView):
    model = studentInformation
    fields = ['firstname','lastname','exam1_marks','exam2_marks']
    
    success_url = reverse_lazy('dataVIZ:listview')
    
class DeleteStudent(DeleteView):
    model = studentInformation
    
    success_url = reverse_lazy('dataVIZ:listview')
    
#------------------------------------------------------------------------------------------------------------------    
    
    
def student_details_page(request, student_id):
   
    stud = studentInformation.objects.get(pk=student_id)
    
    template = loader.get_template('dataVIZ/studentDetails.html')
    
    context = {'student':stud}
    
    return render(request,'dataVIZ/studentDetails.html',context)

    
def post_form_upload(request):
    if request.method == 'POST':
        
        form = studentInformationForm(request.POST)
        
        if form.is_valid():
            firstname = request.POST.get('firstname','')
            lastname = request.POST.get('lastname','')
            exam1_marks = request.POST.get('exam1_marks','')
            exam2_marks = request.POST.get('exam2_marks','')
            
            form_obj = studentInformation(firstname=firstname,
                                         lastname = lastname,
                                         exam1_marks = exam1_marks,
                                         exam2_marks = exam2_marks)
            
            form_obj.save()
            return render(request,'dataVIZ/records.html')
            
    else:
        form = studentInformationForm()

        return render(request,'dataVIZ/insert.html')
    
