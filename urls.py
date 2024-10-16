from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path("AdminLogin", views.AdminLogin, name="AdminLogin"),
	       path("Admin.html", views.Admin, name="Admin"),
	       path("kmeanssvm", views.kmeanssvm, name="kmeanssvm"),
	       path("knnkmeans", views.knnkmeans, name="knnkmeans"),
	       path("nbcnn", views.nbcnn, name="nbcnn"),
	       path("TestSentiment.html", views.TestSentiment, name="TestSentiment"),
	       path("Upload.html", views.Upload, name="Upload"),
	       path("UploadDataset", views.UploadDataset, name="UploadDataset"),
	       path("DetectSentiment", views.DetectSentiment, name="DetectSentiment"),
	       
]