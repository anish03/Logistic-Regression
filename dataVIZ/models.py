from django.db import models
from django.core.urlresolvers import reverse
from django.core.validators import MinValueValidator, MaxValueValidator    

class studentInformation(models.Model):
    firstname = models.CharField(max_length=20)
    lastname = models.CharField(max_length=20)
    exam1_marks = models.IntegerField(validators=[MinValueValidator(0),
                                       MaxValueValidator(100)])
    exam2_marks = models.IntegerField(validators=[MinValueValidator(0),
                                       MaxValueValidator(100)])
    
    def get_absolute_url(self):
        return reverse('dataVIZ:detailview',kwargs={'pk':self.pk})
    
    def __unicode__(self):
        return self.firstname
    

    
