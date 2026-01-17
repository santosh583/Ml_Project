from django.db import models

# Create your models here.

from django.db import models

class News(models.Model):
    text = models.TextField()
    result = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.result} - {self.text[:30]}..."
